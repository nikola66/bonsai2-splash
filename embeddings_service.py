#!/usr/bin/env python3
"""OpenAI-compatible embeddings service backed by MLX (mlx-vlm).

The Splash engine is a causal chat engine: it has no pooling head and no
`/v1/embeddings`. This small, Apple-native service fills that gap with
EmbeddingGemma 2 (text-only 270M) running on the Metal GPU through MLX — no
llama.cpp, no PyTorch. `openai_proxy.py` forwards `POST /v1/embeddings` here and
`ollama_bridge.py` forwards `/api/embed`, so clients talk to one server.

    POST /v1/embeddings   OpenAI embeddings (input: str | [str], dimensions?)
    GET  /health          readiness probe (200 once the model is loaded)
    GET  /status          model id, native dimension, default dimension
    GET  /               one-line info

Dimensions use EmbeddingGemma's Matryoshka (MRL) property: a request may ask for
a smaller vector (128/256/512), which is truncated from the native 768 and
re-normalized to unit length. `BONSAI_EMBED_DIM` sets the default when a request
omits `dimensions`.

Config (environment wins over the git-ignored bonsai.env, which wins over the
defaults — same rule as openai_proxy.py and ollama_bridge.py):

    BONSAI_EMBED_MODEL       MLX repo id (default: text-only 270M conversion)
    BONSAI_EMBED_PORT        loopback port to bind (default 8082)
    BONSAI_EMBED_DIM         default output size when the request omits one
    BONSAI_EMBED_MAX_TOKENS  per-input token cap (default 8192)

Run with an interpreter that has mlx-vlm installed:

    .venv/bin/python embeddings_service.py

The service binds loopback only (the proxy owns the public port); it is never
reachable from the network.
"""

from __future__ import annotations

import base64
import json
import os
import queue
import sys
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

DEMO_DIR = Path(__file__).resolve().parent

# ── Configuration ────────────────────────────────────────────────────────────
# Read bonsai.env directly (not sourced), so a value cannot execute anything;
# the real environment wins, as in openai_proxy.py and ollama_bridge.py.
_ENV_FILE: dict[str, str] = {}
_env_path = DEMO_DIR / "bonsai.env"
if _env_path.is_file():
    for _line in _env_path.read_text().splitlines():
        _line = _line.strip()
        if not _line or _line.startswith("#") or "=" not in _line:
            continue
        _key, _, _val = _line.partition("=")
        _key = _key.strip()
        if _key.replace("_", "").isalnum():
            _ENV_FILE[_key] = _val.strip().strip("'\"")


def _setting(key: str, default: str) -> str:
    """Environment variable wins over bonsai.env, which wins over `default`."""
    return os.environ.get(key) or _ENV_FILE.get(key) or default


MODEL_ID = _setting("BONSAI_EMBED_MODEL", "gahabeen/embeddinggemma-2-text-270m-mlx-bf16")
PORT = int(_setting("BONSAI_EMBED_PORT", "8082"))
DEFAULT_DIM = int(_setting("BONSAI_EMBED_DIM", "768"))
MAX_TOKENS = int(_setting("BONSAI_EMBED_MAX_TOKENS", "8192"))
HOST = "127.0.0.1"

_model = None
_tokenizer = None
_native_dim = 0

# MLX gives each thread its own GPU stream, so the model must be loaded and used
# on ONE thread. Requests arrive on many threads (ThreadingHTTPServer), so they
# hand work to a dedicated worker through this queue instead of touching the
# model directly.
_jobs: "queue.Queue" = queue.Queue()
_ready = threading.Event()
_init_error: BaseException | None = None


def _log(msg: str) -> None:
    sys.stderr.write(f"embeddings: {msg}\n")
    sys.stderr.flush()


def _effective_dim() -> int:
    """The default output size: BONSAI_EMBED_DIM, capped at the native width."""
    if _native_dim and DEFAULT_DIM > _native_dim:
        return _native_dim
    return DEFAULT_DIM or _native_dim


def _load() -> None:
    """Load the MLX embedding model and tokenizer once, at startup."""
    global _model, _tokenizer, _native_dim
    try:
        import mlx.core as mx
        from huggingface_hub import snapshot_download
        from mlx_vlm.embedding_loader import load_embedding_model
        from transformers import AutoTokenizer
    except ImportError as exc:
        _log(
            "mlx-vlm is not installed in this interpreter "
            f"({exc}). Install it with:  .venv/bin/python -m pip install 'mlx-vlm'"
        )
        raise SystemExit(1)

    _log(f"loading {MODEL_ID} (first run downloads ~580 MB from Hugging Face)…")
    t0 = time.monotonic()
    path = Path(snapshot_download(MODEL_ID))
    # The text-only conversion declares vision_config/audio_config = null, so
    # mlx-vlm builds only the text backbone and skips the tower weights.
    _model = load_embedding_model(path)
    _tokenizer = AutoTokenizer.from_pretrained(path)

    text_config = getattr(getattr(_model, "config", None), "text_config", None)
    _native_dim = int(getattr(text_config, "embedding_dim", 0) or 0)
    if not _native_dim:
        # Fall back to what the model actually emits.
        enc = _tokenizer(["probe"], return_tensors="np")
        out = _model(
            mx.array(enc["input_ids"]),
            attention_mask=mx.array(enc["attention_mask"]),
        )
        _native_dim = int(out.text_embeds.shape[-1])
    if DEFAULT_DIM > _native_dim:
        _log(f"BONSAI_EMBED_DIM={DEFAULT_DIM} exceeds the model's {_native_dim}; will cap")
    _log(
        f"ready in {time.monotonic() - t0:.1f}s: {MODEL_ID} "
        f"({_native_dim}-dim, default {_effective_dim()}, max {MAX_TOKENS} tokens)"
    )


def _embed(texts: list[str], dim: int) -> tuple[list[list[float]], int]:
    """Embed a batch of texts -> (unit-length vectors, prompt_tokens).

    `dim` may be smaller than the model's native width: that is Matryoshka
    truncation, which drops the trailing dimensions and re-normalizes.
    Only the worker thread calls this, so the MLX model is never used
    concurrently.
    """
    import numpy as np
    import mlx.core as mx

    enc = _tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=MAX_TOKENS,
        return_tensors="np",
    )
    out = _model(
        mx.array(enc["input_ids"]),
        attention_mask=mx.array(enc["attention_mask"]),
    )
    vectors = np.array(out.text_embeds.tolist(), dtype=np.float32)
    tokens = int(enc["attention_mask"].sum())

    if dim < vectors.shape[-1]:
        vectors = vectors[:, :dim]
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    vectors = vectors / norms
    return vectors.tolist(), tokens


def _submit(texts: list[str], dim: int) -> tuple[list[list[float]], int]:
    """Hand an embedding job to the worker thread and wait for its result."""
    box: dict = {"done": threading.Event()}
    _jobs.put((texts, dim, box))
    box["done"].wait()
    if "error" in box:
        raise box["error"]
    return box["result"]


def _worker() -> None:
    """Owns the MLX model for the life of the process (one GPU stream)."""
    global _init_error
    try:
        _load()
    except BaseException as exc:  # noqa: BLE001 - reported to main()
        _init_error = exc
        _ready.set()
        return
    _ready.set()
    while True:
        job = _jobs.get()
        if job is None:
            return
        texts, dim, box = job
        try:
            box["result"] = _embed(texts, dim)
        except BaseException as exc:  # noqa: BLE001 - surfaced to the client
            box["error"] = exc
        finally:
            box["done"].set()


def _encode_base64(vec: list[float]) -> str:
    import numpy as np

    return base64.b64encode(np.asarray(vec, dtype="<f4").tobytes()).decode()


def _normalize_input(value) -> list[str]:
    if isinstance(value, str):
        if not value.strip():
            raise ValueError("`input` must not be empty")
        return [value]
    if isinstance(value, list):
        if not value:
            raise ValueError("`input` must be a non-empty string or list of strings")
        out = []
        for item in value:
            if not isinstance(item, str):
                raise ValueError(
                    "`input` list items must be strings (token-array input is not supported)"
                )
            out.append(item)
        return out
    raise ValueError("`input` is required and must be a string or list of strings")


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "bonsai-embeddings"
    sys_version = ""

    def log_message(self, fmt, *args):  # our own logging below
        pass

    def log_error(self, fmt, *args):
        _log(f"request error: {fmt % args if args else fmt}")

    def _body(self) -> bytes | None:
        length = self.headers.get("Content-Length")
        if length is None:
            return None
        try:
            n = int(length)
        except ValueError:
            raise ValueError("invalid Content-Length") from None
        return self.rfile.read(n)

    def _send_json(self, status: int, obj: dict) -> None:
        data = json.dumps(obj, ensure_ascii=False).encode()
        self.send_response_only(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(data)

    def _error(self, status: int, message: str, err_type: str = "invalid_request_error") -> None:
        self._send_json(
            status,
            {"error": {"message": message, "type": err_type, "param": None, "code": err_type}},
        )

    # ── routes ────────────────────────────────────────────────────────────────
    def do_GET(self):
        path = self.path.split("?", 1)[0].rstrip("/") or "/"
        if path == "/health":
            if _model is None:
                return self._error(503, "model still loading", "api_error")
            return self._send_json(200, {"status": "ok", "model": MODEL_ID})
        if path == "/status":
            return self._send_json(
                200,
                {
                    "model": MODEL_ID,
                    "dimensions": _native_dim,
                    "default_dimensions": _effective_dim(),
                    "max_input_tokens": MAX_TOKENS,
                    "ready": _model is not None,
                },
            )
        if path == "/":
            return self._send_json(
                200, {"service": "bonsai-embeddings", "model": MODEL_ID, "dimensions": _native_dim}
            )
        return self._error(404, f"unknown endpoint: {path}", "not_found")

    do_HEAD = do_GET

    def do_OPTIONS(self):
        self.send_response_only(204)
        self.send_header("Access-Control-Allow-Origin", self.headers.get("Origin") or "*")
        self.send_header("Access-Control-Allow-Methods", "POST, GET, HEAD, OPTIONS")
        self.send_header(
            "Access-Control-Allow-Headers",
            self.headers.get("Access-Control-Request-Headers") or "Content-Type, Authorization",
        )
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_POST(self):
        path = self.path.split("?", 1)[0].rstrip("/") or "/"
        if path != "/v1/embeddings":
            return self._error(404, f"unknown endpoint: {path}", "not_found")
        try:
            raw = self._body()
        except ValueError as exc:
            return self._error(400, str(exc))
        if raw is None:
            return self._error(400, "request body is required")
        try:
            req = json.loads(raw or b"{}")
        except ValueError:
            return self._error(400, "request body must be valid JSON")
        if not isinstance(req, dict):
            return self._error(400, "request body must be a JSON object")
        if _model is None:
            return self._error(503, "embeddings model is still loading; retry shortly", "api_error")

        try:
            texts = _normalize_input(req.get("input"))
            dim = req.get("dimensions")
            if dim is None:
                dim = _effective_dim()
            elif isinstance(dim, bool) or not isinstance(dim, int) or not (1 <= dim <= _native_dim):
                raise ValueError(f"`dimensions` must be an integer between 1 and {_native_dim}")
            fmt = req.get("encoding_format") or "float"
            if fmt not in ("float", "base64"):
                raise ValueError("`encoding_format` must be 'float' or 'base64'")
        except ValueError as exc:
            return self._error(400, str(exc))

        try:
            vectors, tokens = _submit(texts, dim)
        except Exception as exc:  # noqa: BLE001 - surface the reason to the client
            _log(f"embedding failed: {exc}")
            traceback.print_exc(file=sys.stderr)
            return self._error(500, f"embedding failed: {exc}", "api_error")

        encode = _encode_base64 if fmt == "base64" else (lambda v: v)
        return self._send_json(
            200,
            {
                "object": "list",
                "data": [
                    {"object": "embedding", "index": i, "embedding": encode(v)}
                    for i, v in enumerate(vectors)
                ],
                "model": req.get("model") or MODEL_ID,
                "usage": {"prompt_tokens": tokens, "total_tokens": tokens},
            },
        )


def main() -> None:
    # Load the model on the worker thread (MLX streams are per-thread), then
    # start serving only once it is ready — the launcher polls /health.
    threading.Thread(target=_worker, daemon=True).start()
    _ready.wait()
    if _init_error is not None:
        _log(f"cannot start: {_init_error}")
        raise SystemExit(1)
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    server.daemon_threads = True
    print(
        f"embeddings service listening on http://{HOST}:{PORT} "
        f"(model={MODEL_ID}, dim={_native_dim})",
        flush=True,
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
