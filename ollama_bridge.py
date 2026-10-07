#!/usr/bin/env python3
"""Ollama-native API bridge for the Bonsai 2 Splash server.

Serves Ollama's REST surface and translates to the upstream OpenAI-compatible
Splash server. It has no authentication of its own — bind it to loopback, or put
it behind an authenticating network, before exposing it:

    GET  /api/version, /api/tags, /api/ps, /health
    POST /api/chat, /api/generate  -> /v1/chat/completions (SSE -> NDJSON)
    POST /api/show                 -> model metadata
    POST /api/embed, /api/embeddings -> not served by Splash (501)

    GET/POST /bonsai/context       -> context-window control for the web UI:
    writes .bonsai-ctx (read by bonsai-service.sh), then SIGTERMs splash serve
    so launchd KeepAlive restarts it with the new --max-context.

Run with the demo venv (provides httpx):  .venv/bin/python ollama_bridge.py
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import signal
import subprocess
import uuid
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

import httpx

DEMO_DIR = Path(__file__).resolve().parent

# Shared config (git-ignored bonsai.env, generated from bonsai.env.example) so the
# bridge and the server agree on the bind address and ports. Read directly rather
# than sourced, so a value cannot execute anything; the real environment wins.
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


# Bind to loopback by default. Set BRIDGE_HOST (or BONSAI_HOST in bonsai.env) to
# a LAN/VPN address to reach the bridge from other devices — do that only behind
# an authenticating network or a firewall, because this bridge has no auth.
HOST = _setting("BRIDGE_HOST", _setting("BONSAI_HOST", "127.0.0.1"))
PORT = int(_setting("BRIDGE_PORT", "11434"))
UPSTREAM = _setting("BRIDGE_UPSTREAM", f"http://{HOST}:{_setting('PORT', '8080')}").rstrip("/")
OLLAMA_VERSION = os.environ.get("BRIDGE_OLLAMA_VERSION", "0.5.13")
GGUF_DIR = DEMO_DIR / "models" / "bonsai2-gguf" / "27B"
TIMEOUT = httpx.Timeout(600.0, connect=10.0)

# Context-window control (web UI): persisted here, read by bonsai-service.sh.
CTX_FILE = DEMO_DIR / ".bonsai-ctx"
CTX_MIN = 8192
CTX_MAX = 262144
# int8 KV cache bytes per context token (Splash's default KV format; the old
# llama.cpp demo used FP16 at 65536 — see AGENTS.md).
KV_BYTES_PER_TOKEN = int(os.environ.get("BRIDGE_KV_BYTES_PER_TOKEN", "32768"))

# Ollama request "options" -> OpenAI chat-completions request fields.
OPTION_MAP = {
    "temperature": "temperature",
    "top_k": "top_k",
    "top_p": "top_p",
    "num_predict": "max_tokens",
    "stop": "stop",
    "repeat_penalty": "repeat_penalty",
    "presence_penalty": "presence_penalty",
    "frequency_penalty": "frequency_penalty",
    "min_p": "min_p",
    "tfs_z": "tfs_z",
    "seed": "seed",
}
FINISH_MAP = {
    "stop": "stop",
    "length": "length",
    "tool_calls": "stop",
    "function_call": "stop",
    "content_filter": "error",
}

_info_cache: dict = {"mid": None, "size": 0, "quant": "unknown"}


def _iso_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def gguf_display_info(mid: str) -> dict:
    """Best-effort file size + quant band for display fields (tags/show/ps).
    Reads the local model dir when it still exists; the Splash engine keeps
    its weights in the Hugging Face cache, so this degrades to zeros/unknown."""
    if _info_cache.get("mid") == mid:
        return _info_cache
    size, quant = 0, "unknown"
    try:
        if GGUF_DIR.is_dir():
            candidates = [p for p in sorted(GGUF_DIR.glob("*.gguf")) if "mmproj" not in p.name]
        else:
            candidates = []
        chosen = None
        for p in candidates:
            if mid and (mid in p.stem or p.stem.startswith(mid) or mid in p.name):
                chosen = p
                break
        if chosen is None and candidates:
            chosen = candidates[0]
        if chosen is not None:
            size = chosen.stat().st_size
            for band in ("PQ2_0", "PTQ1_0", "Q8_0", "Q4_0", "F16"):
                if band in chosen.name:
                    quant = band
                    break
    except OSError:
        pass
    if size == 0 or quant == "unknown":
        # No local GGUF dir anymore: derive from the Splash server itself —
        # the band from the served repository id's variant (`OWNER/REPO:PQ2_0`)
        # and the weights size from the engine's memory plan.
        try:
            with _client() as c:
                if quant == "unknown":
                    for uid in upstream_ids(c):
                        tail = uid.rsplit(":", 1)
                        if (
                            len(tail) == 2
                            and tail[1].lower() != "latest"
                            and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", tail[1])
                        ):
                            quant = tail[1]
                            break
                if size == 0:
                    st = c.get(f"{UPSTREAM}/status")
                    size = int(
                        st.json()
                        .get("memory_plan", {})
                        .get("model", {})
                        .get("memory", {})
                        .get("target_weights_bytes", 0)
                    )
        except (httpx.HTTPError, OSError, ValueError, AttributeError, TypeError):
            pass
    _info_cache.update(mid=mid, size=size, quant=quant)
    return _info_cache


def _client() -> httpx.Client:
    return httpx.Client(timeout=TIMEOUT, follow_redirects=True)


def upstream_ids(client: httpx.Client) -> list:
    r = client.get(f"{UPSTREAM}/v1/models")
    r.raise_for_status()
    return [m.get("id", "") for m in r.json().get("data", []) if m.get("id")] or ["model"]


def pick_model(requested, ids: list) -> str:
    """Match the caller's Ollama name (tag optional, case-insensitive) to the
    upstream model id; fall back to the single served model."""
    if requested:
        want = str(requested).split(":")[0].lower()
        for mid in ids:
            low = mid.lower()
            if want == low or want in low or low in want:
                return mid
    return ids[0]


def ollama_name(mid: str) -> str:
    return mid if ":" in mid else f"{mid}:latest"


def primary_model_id(ids: list) -> str:
    """The one id to show in client model lists (tags/ps): prefer the short
    served alias (no `owner/` path) over the full repository id, so lists like
    `ollama list` show a single, clean entry instead of alias + repo id."""
    return next((i for i in ids if i and "/" not in i), ids[0] if ids else "model")


def details(mid: str) -> dict:
    info = gguf_display_info(mid)
    return {
        "parent_model": "",
        "format": "gguf",
        "family": "qwen3",
        "families": ["qwen3"],
        "parameter_size": "27B",
        "quantization_level": info["quant"],
    }


def healthy() -> bool:
    try:
        with httpx.Client(timeout=httpx.Timeout(3.0)) as c:
            return c.get(f"{UPSTREAM}/health").status_code == 200
    except httpx.HTTPError:
        return False


def _splash_pids() -> list:
    """PIDs of the Splash HTTP server (`python -m server.server ...`) this
    bridge translates to: matched on its cmdline carrying the same --host and
    --port as UPSTREAM, so a context restart never signals an unrelated Splash
    instance. SIGTERM to this process stops the whole engine tree (it owns the
    serve-native child that holds the weights); the `splash serve` wrapper does
    NOT forward signals to it."""
    up = urlparse(UPSTREAM)
    host = re.escape(up.hostname or "")
    port = up.port or (443 if up.scheme == "https" else 80)
    # Arg order in the cmdline differs (`--port 8080 --host=IP`); accept both.
    # Splash also omits --host entirely when it is the default (127.0.0.1),
    # so only require the host when upstream points somewhere else.
    if (up.hostname or "") in ("127.0.0.1", "localhost"):
        pat = rf"server\.server .*--port[= ]{port}"
    else:
        pat = (
            rf"server\.server .*--host[= ]{host}.*--port[= ]{port}"
            rf"|server\.server .*--port[= ]{port}.*--host[= ]{host}"
        )
    try:
        out = subprocess.run(
            ["pgrep", "-f", pat],
            capture_output=True, text=True, timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return []
    if out.returncode != 0:
        return []
    pids = []
    for tok in out.stdout.split():
        try:
            pids.append(int(tok))
        except ValueError:
            continue
    return pids


def _total_ram_bytes() -> int:
    try:
        out = subprocess.run(
            ["sysctl", "-n", "hw.memsize"],
            capture_output=True, text=True, timeout=5,
        )
        if out.returncode == 0:
            return int(out.stdout.strip())
    except (OSError, ValueError, subprocess.TimeoutExpired):
        pass
    return 0


def _model_rss_bytes() -> int | None:
    pids = _splash_pids()
    if not pids:
        return None
    # The weights live in the serve-native child(ren) of each server process.
    for pid in list(pids):
        try:
            out = subprocess.run(
                ["pgrep", "-P", str(pid)],
                capture_output=True, text=True, timeout=5,
            )
            if out.returncode == 0:
                pids.extend(int(t) for t in out.stdout.split() if t.isdigit())
        except (OSError, subprocess.TimeoutExpired):
            pass
    try:
        out = subprocess.run(
            ["ps", "-o", "rss=", "-p", ",".join(str(p) for p in pids)],
            capture_output=True, text=True, timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if out.returncode != 0:
        return None
    rss = []
    for tok in out.stdout.split():
        try:
            rss.append(int(tok))
        except ValueError:
            continue
    # max() avoids double-counting the brief old+new overlap during a restart
    return max(rss) * 1024 if rss else None


def _read_ctx_file() -> int | None:
    try:
        raw = CTX_FILE.read_text().strip()
    except OSError:
        return None
    return int(raw) if raw.isdigit() else None


def _upstream_ctx() -> int | None:
    """The engine's effective context limit (Splash /status), or None while it
    is down (mid-restart)."""
    try:
        with _client() as c:
            r = c.get(f"{UPSTREAM}/status")
            if r.status_code == 200:
                n = (r.json() or {}).get("maximum_context_tokens")
                return int(n) if isinstance(n, int) else None
    except (httpx.HTTPError, ValueError):
        pass
    return None


# ── message / tool-call conversion ──────────────────────────────────────────

def to_openai_tool_calls(tcs) -> list:
    out = []
    for tc in tcs or []:
        fn = tc.get("function") or {}
        args = fn.get("arguments", {})
        if not isinstance(args, str):
            args = json.dumps(args)
        out.append({
            "id": tc.get("id") or f"call_{uuid.uuid4().hex[:24]}",
            "type": "function",
            "function": {"name": fn.get("name", ""), "arguments": args},
        })
    return out


def to_ollama_tool_calls(tcs) -> list:
    out = []
    for tc in tcs or []:
        fn = tc.get("function") or {}
        args = fn.get("arguments", "{}")
        if isinstance(args, str):
            try:
                args = json.loads(args) if args else {}
            except ValueError:
                args = {"_raw": args}
        out.append({"function": {"name": fn.get("name", ""), "arguments": args}})
    return out


def to_openai_messages(messages) -> list:
    out = []
    for m in messages or []:
        role = m.get("role") or "assistant"
        content = m.get("content")
        if isinstance(content, list):
            parts = content
        else:
            text = "" if content is None else (content if isinstance(content, str) else str(content))
            parts = [{"type": "text", "text": text}]
        for img in m.get("images") or []:
            if isinstance(img, str):
                b64 = img.split(",", 1)[-1] if img.startswith("data:") else img
                parts.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}})
        nm = {"role": role}
        first = parts[0] if parts else {}
        if len(parts) == 1 and (first.get("type", "text") == "text") and "text" in first:
            nm["content"] = first["text"]
        else:
            nm["content"] = parts
        if m.get("tool_calls"):
            nm["tool_calls"] = to_openai_tool_calls(m["tool_calls"])
        tool_name = m.get("tool_name") or m.get("name")
        if role == "tool" and tool_name:
            nm["name"] = tool_name
        out.append(nm)
    return out


def build_body(req: dict, stream: bool) -> dict:
    """Ollama request -> OpenAI chat/completions body (model/messages added by caller)."""
    body: dict = {"stream": stream}
    if stream:
        body["stream_options"] = {"include_usage": True}
    opts = req.get("options") or {}
    for key, val in opts.items():
        field = OPTION_MAP.get(key)
        if field is None or val is None:
            continue
        if field == "max_tokens" and isinstance(val, int) and val <= 0:
            continue  # Ollama: <= 0 means unlimited
        body[field] = val
    for key in ("temperature", "top_p", "max_tokens", "stop"):
        if req.get(key) is not None and key not in body:
            body[key] = req[key]
    if req.get("tools"):
        body["tools"] = req["tools"]
    fmt = req.get("format")
    if fmt == "json":
        body["response_format"] = {"type": "json_object"}
    elif isinstance(fmt, dict):
        body["response_format"] = {
            "type": "json_schema",
            "json_schema": {"name": "response", "schema": fmt},
        }
    # Ollama's think: false/None-ish off, "low"/"medium"/"high" = effort level.
    # Absent or true = the server default (thinking on), as Ollama specifies.
    think = req.get("think")
    if think is not None:
        if think is False:
            body["reasoning_effort"] = "none"
        elif isinstance(think, str) and think.lower() in ("low", "medium", "high"):
            body["reasoning_effort"] = think.lower()
    return body


def durations_ns(t: dict) -> dict:
    """Engine timings (llama-server-style keys) -> Ollama ns durations."""
    out = {}

    def secs(base):
        if f"{base}_ms" in t:
            return float(t[f"{base}_ms"]) / 1000.0
        if f"{base}_t" in t:
            return float(t[f"{base}_t"])
        return None

    prompt = secs("prompt")
    predicted = secs("predicted")
    if prompt is not None:
        out["prompt_eval_duration"] = int(prompt * 1e9)
    if predicted is not None:
        out["eval_duration"] = int(predicted * 1e9)
    if prompt is not None and predicted is not None:
        out["total_duration"] = int((prompt + predicted) * 1e9)
    return out


# ── HTTP handler ────────────────────────────────────────────────────────────

class Handler(BaseHTTPRequestHandler):
    server_version = "ollama/0.5.13"
    protocol_version = "HTTP/1.0"  # close-delimited NDJSON streaming

    # plumbing

    def _body(self):
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b"{}"
        try:
            parsed = json.loads(raw or b"{}")
            return parsed if isinstance(parsed, dict) else None
        except ValueError:
            return None

    def _send_json(self, obj, status=200, ndjson=False, cors=False):
        data = (json.dumps(obj, ensure_ascii=False) + ("" if ndjson else "")).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/x-ndjson" if ndjson else "application/json")
        if cors:
            self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _error(self, status, message, cors=False):
        self._send_json({"error": message}, status, cors=cors)

    def _ndjson_line(self, obj):
        self.wfile.write((json.dumps(obj, ensure_ascii=False) + "\n").encode())
        self.wfile.flush()

    # CORS preflight for /bonsai/* (the web UI runs on another port)

    def do_OPTIONS(self):
        path = self.path.split("?", 1)[0]
        if not path.startswith("/bonsai/"):
            return self._error(404, "not found")
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "content-type")
        self.send_header("Content-Length", "0")
        self.end_headers()

    # GET

    def do_GET(self):
        path = self.path.split("?", 1)[0]
        try:
            if path == "/health":
                return self._send_json({"status": "ok"})
            if path == "/bonsai/context":
                return self._handle_context_get()
            if path == "/api/version":
                return self._send_json({"version": OLLAMA_VERSION})
            if path == "/api/tags":
                # One entry: the served alias (both ids in /v1/models are the
                # same model — alias + repository id).
                with _client() as c:
                    mid = primary_model_id(upstream_ids(c))
                info = gguf_display_info(mid)
                models = [{
                    "name": ollama_name(mid),
                    "model": ollama_name(mid),
                    "modified_at": _iso_now(),
                    "size": info["size"],
                    "digest": hashlib.sha256(mid.encode()).hexdigest(),
                    "details": details(mid),
                }]
                return self._send_json({"models": models})
            if path == "/api/ps":
                models = []
                if healthy():
                    with _client() as c:
                        mid = primary_model_id(upstream_ids(c))
                    info = gguf_display_info(mid)
                    models.append({
                        "name": ollama_name(mid),
                        "model": ollama_name(mid),
                        "size": info["size"],
                        "expires_at": (datetime.now(timezone.utc) + timedelta(minutes=30)).strftime(
                            "%Y-%m-%dT%H:%M:%S.%fZ"),
                        "created_at": _iso_now(),
                        "details": details(mid),
                    })
                return self._send_json({"models": models})
            if path.startswith("/api/"):
                return self._error(404, f"unknown endpoint: {path}")
            return self._error(404, "not found")
        except httpx.HTTPStatusError as exc:
            return self._error(500, f"upstream returned {exc.response.status_code}")
        except (httpx.HTTPError, OSError) as exc:
            return self._error(500, f"splash server not ready: {exc}")

    # POST

    def do_POST(self):
        path = self.path.split("?", 1)[0]
        req = self._body()
        if req is None:
            return self._error(400, "invalid JSON body", cors=path.startswith("/bonsai/"))
        try:
            if path == "/bonsai/context":
                return self._handle_context_post(req)
            if path == "/api/chat":
                return self._handle_chat(req)
            if path == "/api/generate":
                return self._handle_generate(req)
            if path == "/api/show":
                return self._handle_show(req)
            if path in ("/api/embed", "/api/embeddings"):
                return self._handle_embed(req, path)
            if path in ("/api/pull", "/api/delete", "/api/copy", "/api/create", "/api/blobs", "/api/run"):
                return self._error(501, "model management is handled on the host")
            if path.startswith("/api/"):
                return self._error(404, f"unknown endpoint: {path}")
            return self._error(404, "not found")
        except httpx.HTTPStatusError as exc:
            text = exc.response.text[:2000]
            code = exc.response.status_code
            return self._error(code if 400 <= code < 500 else 500, f"upstream {code}: {text}")
        except (httpx.HTTPError, OSError) as exc:
            return self._error(500, f"splash server not ready: {exc}")

    def _handle_context_get(self):
        return self._send_json({
            "n_ctx": _upstream_ctx(),
            "file_n_ctx": _read_ctx_file(),
            "total_ram_bytes": _total_ram_bytes(),
            "model_rss_bytes": _model_rss_bytes(),
            "model_file_bytes": gguf_display_info("").get("size", 0),
            "kv_bytes_per_token": KV_BYTES_PER_TOKEN,
        }, cors=True)

    def _handle_context_post(self, req):
        n = req.get("n_ctx")
        if isinstance(n, bool) or not isinstance(n, int) or not (CTX_MIN <= n <= CTX_MAX):
            return self._error(
                400, f"n_ctx must be an integer between {CTX_MIN} and {CTX_MAX}", cors=True)
        try:
            CTX_FILE.write_text(f"{n}\n")
        except OSError as exc:
            return self._error(500, f"cannot write {CTX_FILE.name}: {exc}", cors=True)
        # Respond first, then SIGTERM: launchd KeepAlive restarts with the new
        # --max-context (bonsai-service.sh re-reads .bonsai-ctx).
        pids = _splash_pids()
        self._send_json({"status": "restarting" if pids else "saved", "n_ctx": n}, cors=True)
        for pid in pids:
            try:
                os.kill(pid, signal.SIGTERM)
            except OSError:
                pass

    def _resolve(self, req) -> tuple:
        with _client() as c:
            mid = pick_model(req.get("model") or req.get("name"), upstream_ids(c))
        return mid

    def _handle_chat(self, req):
        mid = self._resolve(req)
        stream = bool(req.get("stream", True))
        body = build_body(req, stream)
        body["model"] = mid
        body["messages"] = to_openai_messages(req.get("messages"))
        if stream:
            return self._stream(body, mid, mode="chat")
        with _client() as c:
            r = c.post(f"{UPSTREAM}/v1/chat/completions", json=body)
            r.raise_for_status()
            data = r.json()
        choice = (data.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        out_msg = {"role": "assistant", "content": message.get("content") or ""}
        reasoning = message.get("reasoning_content") or message.get("reasoning")
        if reasoning:
            out_msg["reasoning"] = reasoning
        if message.get("tool_calls"):
            out_msg["tool_calls"] = to_ollama_tool_calls(message["tool_calls"])
        final = {
            "model": ollama_name(mid),
            "created_at": _iso_now(),
            "message": out_msg,
            "done": True,
            "done_reason": FINISH_MAP.get(choice.get("finish_reason") or "", "stop"),
        }
        usage = data.get("usage") or {}
        if usage.get("prompt_tokens") is not None:
            final["prompt_eval_count"] = usage["prompt_tokens"]
        if usage.get("completion_tokens") is not None:
            final["eval_count"] = usage["completion_tokens"]
        final.update(durations_ns(data.get("timings") or {}))
        return self._send_json(final)

    def _handle_generate(self, req):
        mid = self._resolve(req)
        stream = bool(req.get("stream", True))
        body = build_body(req, stream)
        body["model"] = mid
        messages = []
        if req.get("system"):
            messages.append({"role": "system", "content": req["system"]})
        messages.append({
            "role": "user",
            "content": req.get("prompt") or "",
            "images": req.get("images") or [],
        })
        body["messages"] = to_openai_messages(messages)
        if stream:
            return self._stream(body, mid, mode="generate")
        with _client() as c:
            r = c.post(f"{UPSTREAM}/v1/chat/completions", json=body)
            r.raise_for_status()
            data = r.json()
        choice = (data.get("choices") or [{}])[0]
        text = (choice.get("message") or {}).get("content") or ""
        final = {
            "model": ollama_name(mid),
            "created_at": _iso_now(),
            "response": text,
            "done": True,
            "done_reason": FINISH_MAP.get(choice.get("finish_reason") or "", "stop"),
        }
        usage = data.get("usage") or {}
        if usage.get("prompt_tokens") is not None:
            final["prompt_eval_count"] = usage["prompt_tokens"]
        if usage.get("completion_tokens") is not None:
            final["eval_count"] = usage["completion_tokens"]
        final.update(durations_ns(data.get("timings") or {}))
        return self._send_json(final)

    def _stream(self, body, mid, mode):
        name = ollama_name(mid)
        created = _iso_now()
        usage: dict = {}
        finish_reason = None
        # OpenAI streams tool calls as fragments (id/name in the first chunk,
        # arguments split across many). Ollama clients expect ONE message with
        # the complete call — accumulate here, emit after the stream ends.
        tc_frags: dict = {}
        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson")
        self.end_headers()
        try:
            with _client() as c:
                with c.stream("POST", f"{UPSTREAM}/v1/chat/completions", json=body) as r:
                    if r.status_code >= 400:
                        text = r.read().decode(errors="replace")[:1000]
                        return self._ndjson_line({"error": f"upstream {r.status_code}: {text}"})
                    for line in r.iter_lines():
                        if not line.startswith("data:"):
                            continue
                        payload = line[5:].strip()
                        if payload == "[DONE]":
                            break
                        try:
                            chunk = json.loads(payload)
                        except ValueError:
                            continue
                        if isinstance(chunk.get("usage"), dict):
                            usage.update(chunk["usage"])
                        for choice in chunk.get("choices") or []:
                            if choice.get("finish_reason"):
                                finish_reason = choice["finish_reason"]
                            delta = choice.get("delta") or {}
                            text = delta.get("content")
                            reasoning = delta.get("reasoning_content") or delta.get("reasoning")
                            tool_calls = delta.get("tool_calls") or []
                            for tc in tool_calls:
                                acc = tc_frags.setdefault(
                                    tc.get("index", 0), {"id": None, "name": "", "arguments": ""})
                                if tc.get("id"):
                                    acc["id"] = tc["id"]
                                fn = tc.get("function") or {}
                                if fn.get("name"):
                                    acc["name"] += fn["name"]
                                if fn.get("arguments"):
                                    acc["arguments"] += fn["arguments"]
                            if mode == "chat":
                                if text or reasoning:
                                    msg = {"role": "assistant", "content": text or ""}
                                    if reasoning:
                                        msg["reasoning"] = reasoning
                                    self._ndjson_line(
                                        {"model": name, "created_at": created, "message": msg, "done": False})
                            else:
                                if text:
                                    self._ndjson_line(
                                        {"model": name, "created_at": created, "response": text, "done": False})
        except (httpx.HTTPError, OSError) as exc:
            return self._ndjson_line({"error": f"stream failed: {exc}"})
        if mode == "chat" and tc_frags:
            assembled = to_ollama_tool_calls([
                {"id": acc["id"], "type": "function",
                 "function": {"name": acc["name"], "arguments": acc["arguments"]}}
                for _, acc in sorted(tc_frags.items())
            ])
            self._ndjson_line({
                "model": name, "created_at": created,
                "message": {"role": "assistant", "content": "", "tool_calls": assembled},
                "done": False,
            })
        final = {
            "model": name,
            "created_at": created,
            "done": True,
        }
        if mode == "chat":
            final["message"] = {"role": "assistant", "content": ""}
        else:
            final["response"] = ""
        if finish_reason:
            final["done_reason"] = FINISH_MAP.get(finish_reason, "stop")
        if usage.get("prompt_tokens") is not None:
            final["prompt_eval_count"] = usage["prompt_tokens"]
        if usage.get("completion_tokens") is not None:
            final["eval_count"] = usage["completion_tokens"]
        return self._ndjson_line(final)

    def _handle_show(self, req):
        mid = self._resolve(req)
        info = gguf_display_info(mid)
        return self._send_json({
            "modelfile": f"FROM {ollama_name(mid)}",
            "details": details(mid),
            "capabilities": ["completion", "vision", "tools"],
            "parameters": "temperature 1.0\ntop_k 20\ntop_p 0.95",
            "template": "{{ .System }}\n{{ range .Messages }}{{ .Role }}: {{ .Content }}\n{{ end }}",
            "system": "",
            "model_info": {
                "general.architecture": "qwen3",
                "general.context_length": 65536,
                "general.file_type": info["quant"],
            },
        })

    def _handle_embed(self, req, path):
        # The Splash engine has no embeddings endpoint (the old llama.cpp
        # server had /v1/embeddings); answer clearly instead of proxying.
        return self._error(501, "embeddings are not served by the Splash engine")


def main():
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    server.daemon_threads = True
    print(f"ollama bridge listening on http://{HOST}:{PORT} -> {UPSTREAM}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
