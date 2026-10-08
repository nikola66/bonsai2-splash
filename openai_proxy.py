#!/usr/bin/env python3
"""OpenAI-compatible normalization proxy for the Bonsai Splash server.

Sits between clients and the Splash engine so this repo can fix API
behaviour the engine itself does not implement:

  * **Loopback is always served.** The engine binds a single address, so
    configuring a VPN address (BONSAI_HOST) used to make 127.0.0.1 unreachable
    even though every URL in the docs says localhost. The proxy binds the
    configured address *and* 127.0.0.1; the engine binds loopback only.
  * **`model` is required** on the OpenAI text routes (/v1/chat/completions,
    /v1/completions, /v1/responses): a request without a non-empty string model
    is refused with a 400 invalid_request_error instead of being served
    silently. The bundled chat page posts without one, so on the way out the
    proxy injects the served model id into its JavaScript; if that ever stops
    matching the engine's page, enforcement backs off (with a warning) rather
    than break the UI.
  * **Assistant `content` is never null.** When finish_reason is "length" and
    only reasoning tokens were produced, the engine answers content: null,
    which crashes clients that do message.content.strip(). The proxy answers
    "" instead (null stays legitimate when tool_calls are present, as OpenAI
    defines).

Everything else passes through byte-for-byte: the chat page, /v1/models,
/health, /ready, /status, SSE streams (including tool-call deltas), uploads,
error bodies, CORS and API-key headers.

Config (same precedence as ollama_bridge.py: environment wins over the
git-ignored bonsai.env, which wins over defaults):

    BONSAI_HOST          public bind address (default 127.0.0.1)
    PORT                 public port (default 8080)
    BONSAI_ENGINE_PORT   engine port (default PORT + 1, loopback only)

Started and stopped by scripts/start_llama_server.sh. Standard library only.
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
import traceback
from http.client import HTTPConnection, HTTPException
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

DEMO_DIR = Path(__file__).resolve().parent

# ── Configuration ────────────────────────────────────────────────────────────
# Read bonsai.env directly (not sourced), so a value cannot execute anything;
# the real environment wins, as in ollama_bridge.py.
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


HOST = _setting("BONSAI_HOST", "127.0.0.1")
PORT = int(_setting("PORT", "8080"))
ENGINE_PORT = int(_setting("BONSAI_ENGINE_PORT", str(PORT + 1)))
ENGINE = ("127.0.0.1", ENGINE_PORT)

# OpenAI requires `model` on its text routes; the engine accepts these
# requests without one (only its Anthropic route validates the field).
MODEL_REQUIRED = {"/v1/chat/completions", "/v1/completions", "/v1/responses"}

# The bundled chat page's request body (server/chat.html) — the one client
# that legitimately posts without a model field.
PAGE_PATTERN = b"body: JSON.stringify({messages:"

# Headers that describe a single hop and are re-framed by this proxy.
# Expect is dropped because the proxy already answers 100-continue itself;
# Accept-Encoding is dropped so upstream bodies arrive identity-coded and can
# be inspected (SSE, the chat page) — everything here is loopback anyway.
HOP_BY_HOP = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
    "expect",
    "accept-encoding",
    "content-length",
}

# Generous: images and PDFs ride along as base64 inside JSON.
MAX_BODY = 1 << 30

ENGINE_TIMEOUT = 3600.0  # long-context prefill can exceed three minutes

_lock = threading.Lock()
# Chat-page patch state: patchable None = unknown, True/False = decided.
# The proxy restarts with the engine, so caching for the process lifetime is
# enough to track one engine version.
_page = {"patchable": None, "model": None}
_page_warned = False


def _log(msg: str) -> None:
    sys.stderr.write(f"openai-proxy: {msg}\n")
    sys.stderr.flush()


class _BadRequest(Exception):
    pass


class _TooLarge(Exception):
    pass


def _read_exactly(rfile, n: int) -> bytes:
    out = bytearray()
    while len(out) < n:
        block = rfile.read(n - len(out))
        if not block:
            break
        out += block
    return bytes(out)


def _read_body(handler) -> bytes | None:
    """The request body, re-framed: None when the request carries no body.

    Both Content-Length and chunked bodies are fully buffered — the engine
    parses JSON anyway, and buffering is what lets the proxy validate and
    normalize requests before they reach the engine.
    """
    te = (handler.headers.get("Transfer-Encoding") or "").lower()
    if "chunked" in te:
        chunks: list[bytes] = []
        total = 0
        while True:
            size_line = handler.rfile.readline(MAX_BODY + 2)
            if not size_line:
                raise _BadRequest("truncated chunked body")
            size_txt = size_line.split(b";", 1)[0].strip()
            try:
                size = int(size_txt, 16)
            except ValueError:
                raise _BadRequest("invalid chunk size") from None
            if size == 0:
                # Trailer section, ended by an empty line.
                while True:
                    line = handler.rfile.readline(65537)
                    if line in (b"\r\n", b"\n", b""):
                        break
                break
            total += size
            if total > MAX_BODY:
                raise _TooLarge()
            chunk = _read_exactly(handler.rfile, size)
            if len(chunk) != size:
                raise _BadRequest("truncated chunked body")
            chunks.append(chunk)
            if handler.rfile.read(2) not in (b"\r\n", b"\n"):
                raise _BadRequest("invalid chunk framing")
        return b"".join(chunks)
    cl = handler.headers.get("Content-Length")
    if cl is None:
        return None
    try:
        n = int(cl)
    except ValueError:
        raise _BadRequest("invalid Content-Length") from None
    if n > MAX_BODY:
        raise _TooLarge()
    body = _read_exactly(handler.rfile, n)
    if len(body) != n:
        raise _BadRequest("truncated request body")
    return body


def _error_body(message: str, err_type: str = "invalid_request_error") -> bytes:
    return json.dumps(
        {"error": {"message": message, "type": err_type, "code": err_type}}
    ).encode()


def _engine_get(path: str, timeout: float = 10.0) -> tuple[int | None, bytes | None]:
    """A GET against the engine; (None, None) when it is unreachable."""
    conn = HTTPConnection(ENGINE[0], ENGINE[1], timeout=timeout)
    try:
        conn.request("GET", path)
        resp = conn.getresponse()
        return resp.status, resp.read()
    except (OSError, HTTPException):
        return None, None
    finally:
        conn.close()


def _page_model_id() -> str | None:
    """The served model id to inject into the chat page (short alias first)."""
    status, data = _engine_get("/v1/models")
    if status != 200 or not data:
        return None
    try:
        ids = [m.get("id") for m in json.loads(data).get("data", []) if m.get("id")]
    except (ValueError, AttributeError, TypeError):
        return None
    if not ids:
        return None
    return next((i for i in ids if "/" not in i), ids[0])


def _ensure_patch_state() -> str | None:
    """The model id when the chat page can be made spec-compliant (its
    request can carry the injected model field), None when the engine is
    unreachable (transient, retried) or the page no longer matches (cached:
    enforcement backs off so the UI keeps works). Probes the engine; the
    handler's own copy of the page avoids this call on a normal page load."""
    global _page_warned
    with _lock:
        if _page["patchable"] is not None:
            return _page["model"] if _page["patchable"] else None
    status, data = _engine_get("/")
    if status != 200 or not data:
        return None  # transient: engine still starting; do not cache
    if PAGE_PATTERN not in data:
        with _lock:
            _page["patchable"] = False
        if not _page_warned:
            _page_warned = True
            _log(
                "the chat page no longer matches the model-injection pattern; "
                "not enforcing the required model field (normalization is "
                "disabled until the pattern is updated)"
            )
        return None
    mid = _page_model_id()
    if not mid:
        return None  # transient: do not cache
    with _lock:
        _page["patchable"] = True
        _page["model"] = mid
    return mid


def _patch_page(data: bytes, mid: str) -> bytes:
    repl = (
        b"body: JSON.stringify({model:"
        + json.dumps(mid).encode()
        + b", messages:"
    )
    return data.replace(PAGE_PATTERN, repl)


def _normalize_chat(data: bytes) -> bytes:
    """content: null -> "" when the message carries no tool calls."""
    try:
        obj = json.loads(data)
    except ValueError:
        return data
    if not isinstance(obj, dict):
        return data
    choices = obj.get("choices")
    if not isinstance(choices, list):
        return data
    changed = False
    for choice in choices:
        if not isinstance(choice, dict):
            continue
        message = choice.get("message")
        if (
            isinstance(message, dict)
            and message.get("content") is None
            and not message.get("tool_calls")
        ):
            message["content"] = ""
            changed = True
    if not changed:
        return data
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":")).encode()


class ProxyHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "bonsai-openai-proxy"
    sys_version = ""

    # ── plumbing ─────────────────────────────────────────────────────────────
    def log_message(self, fmt, *args):  # our own logging below
        pass

    def log_error(self, fmt, *args):
        _log(f"request error: {fmt % args if args else fmt}")

    def _send(
        self,
        status: int,
        body: bytes,
        headers: list[tuple[str, str]] | None = None,
        keep_alive: bool = True,
    ) -> None:
        self._started = True
        self.send_response_only(status)
        for key, value in headers or ():
            self.send_header(key, value)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)
        if not keep_alive:
            self.close_connection = True

    def _send_json(self, status: int, body: bytes) -> None:
        self._send(status, body, [("Content-Type", "application/json")])

    # ── chat page: inject the served model id so it satisfies the API ───────
    def _maybe_patch_page(self, data: bytes, ctype: str) -> bytes:
        if "text/html" not in ctype or PAGE_PATTERN not in data:
            return data
        with _lock:
            known, mid = _page["patchable"], _page["model"]
        if known is False:
            return data  # engine upgraded; warned once by the probe
        if known is None:
            # This very response matched: only the model id is missing.
            mid = _page_model_id()
            if not mid:
                return data  # transient: serve unpatched this once
            with _lock:
                _page["patchable"] = True
                _page["model"] = mid
        return _patch_page(data, mid)

    # ── the proxy itself ─────────────────────────────────────────────────────
    def _relay(self) -> None:
        self._started = False
        conn: HTTPConnection | None = None
        t0 = time.monotonic()
        method = self.command
        path = urlsplit(self.path).path
        try:
            try:
                body = _read_body(self)
            except _BadRequest as exc:
                self._send_json(400, _error_body(str(exc)))
                self._log_line(path, 400, t0, "bad request framing")
                return
            except _TooLarge:
                # The body was not read: the connection can no longer be
                # framed, so answer and close.
                self._send_json(
                    413,
                    _error_body(
                        f"request body exceeds the proxy limit of {MAX_BODY} bytes"
                    ),
                    keep_alive=False,
                )
                self._log_line(path, 413, t0, "body too large")
                return

            # Parse the body only where the proxy gates or normalizes it.
            parsed = None
            if body and path in MODEL_REQUIRED:
                try:
                    parsed = json.loads(body)
                except ValueError:
                    parsed = None

            # 1) OpenAI requires a non-empty string `model`.
            if path in MODEL_REQUIRED and isinstance(parsed, dict):
                model = parsed.get("model")
                if not (isinstance(model, str) and model):
                    if _ensure_patch_state() is not None:
                        message = (
                            "model is required"
                            if model is None
                            else "model must be a non-empty string"
                        )
                        self._send_json(400, _error_body(message))
                        self._log_line(path, 400, t0, "missing model")
                        return
                    # Chat page not patchable: serve as the engine would.

            stream = isinstance(parsed, dict) and bool(parsed.get("stream"))

            # Headers, re-framed for one hop; the client's Host is forwarded
            # verbatim (the engine validates it) and Accept-Encoding is
            # dropped so bodies arrive identity-coded.
            fwd: dict[str, str] = {}
            for key, value in self.headers.items():
                if key.lower() in HOP_BY_HOP:
                    continue
                fwd[key] = value
            if stream:
                # SSE: raw-pipe the response, so ask the engine to close after
                # it — that is how the pipe knows the body ended.
                fwd["Connection"] = "close"

            conn = HTTPConnection(ENGINE[0], ENGINE[1], timeout=ENGINE_TIMEOUT)
            try:
                conn.request(method, self.path, body=body, headers=fwd)
                resp = conn.getresponse()
            except (OSError, HTTPException) as exc:
                self._send_json(
                    502,
                    _error_body(
                        f"Bonsai engine unavailable on port {ENGINE_PORT}: {exc}",
                        "api_error",
                    ),
                )
                self._log_line(path, 502, t0, "engine unavailable")
                return
            self._started = True  # headers will be forwarded below

            if stream:
                self._pipe(resp)
                self._log_line(path, resp.status, t0, "streamed")
                return

            if method == "HEAD":
                self.send_response_only(resp.status, resp.reason)
                for key, value in resp.headers.items():
                    if key.lower() in ("connection", "keep-alive"):
                        continue
                    self.send_header(key, value)
                self.end_headers()
                self.close_connection = True
                self._log_line(path, resp.status, t0)
                return

            data = resp.read()
            ctype = resp.getheader("Content-Type", "")
            encoded = resp.getheader("Content-Encoding")
            if resp.status == 200 and not encoded:
                if "text/html" in ctype:
                    data = self._maybe_patch_page(data, ctype)
                elif path == "/v1/chat/completions" and "json" in ctype:
                    # 3) content: null -> "" when no tool calls follow it.
                    data = _normalize_chat(data)

            hop = ("connection", "keep-alive", "transfer-encoding", "content-length")
            headers = [
                (k, v)
                for k, v in resp.headers.items()
                if k.lower() not in hop
            ]
            self.send_response_only(resp.status, resp.reason)
            for key, value in headers:
                self.send_header(key, value)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            self._log_line(path, resp.status, t0)
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception:
            _log(f"internal error while proxying {method} {self.path}")
            traceback.print_exc(file=sys.stderr)
            if not self._started:
                try:
                    self._send_json(
                        500,
                        _error_body("proxy internal error", "api_error"),
                        keep_alive=False,
                    )
                except OSError:
                    pass
        finally:
            if conn is not None:
                try:
                    conn.close()
                except OSError:
                    pass

    def _pipe(self, resp) -> None:
        """Forward an SSE response verbatim: status line, headers and raw
        body bytes (chunk framing included), relying on the engine closing
        the connection after it (we requested Connection: close)."""
        self.send_response_only(resp.status, resp.reason)
        for key, value in resp.headers.items():
            if key.lower() in ("connection", "keep-alive"):
                continue
            self.send_header(key, value)
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True
        if self.command == "HEAD" or resp.fp is None:
            return
        fp = resp.fp
        while True:
            # read1: at most one underlying read, so SSE tokens are forwarded
            # as they arrive instead of waiting for a full buffer.
            block = fp.read1(65536)
            if not block:
                break
            self.wfile.write(block)

    def _log_line(self, path: str, status: int, t0: float, note: str = "") -> None:
        # The chat API and every failure are worth a line; routine health and
        # static GETs are not (clients poll them).
        if status < 400 and path not in MODEL_REQUIRED:
            return
        ms = (time.monotonic() - t0) * 1000
        suffix = f" ({note})" if note else ""
        _log(f"{status} {path} {ms:.0f}ms{suffix}")

    # ── methods ──────────────────────────────────────────────────────────────
    def _dispatch(self) -> None:
        try:
            self._relay()
        except (BrokenPipeError, ConnectionResetError):
            pass

    do_GET = do_HEAD = do_POST = do_PUT = do_DELETE = do_PATCH = do_OPTIONS = (
        _dispatch
    )


def _bind_addresses() -> list[str]:
    """Addresses to serve. A concrete BONSAI_HOST (e.g. a VPN address) is
    served *next to* loopback — the engine binds one address only, which is
    how a VPN-only bind used to strand every localhost URL. 0.0.0.0 already
    covers loopback."""
    host = HOST.strip()
    if host == "0.0.0.0":
        return [host]
    if host == "localhost" or host.startswith("127."):
        return [host]
    return [host, "127.0.0.1"]


def main() -> None:
    if PORT == ENGINE_PORT:
        _log(
            f"PORT and BONSAI_ENGINE_PORT are both {PORT}; "
            "give the engine its own port (default PORT + 1)"
        )
        raise SystemExit(1)

    binds = _bind_addresses()
    servers: dict[str, ThreadingHTTPServer] = {}
    pending: list[str] = []
    for addr in binds:
        try:
            srv = ThreadingHTTPServer((addr, PORT), ProxyHandler)
        except OSError as exc:
            _log(f"cannot bind {addr}:{PORT} ({exc})")
            pending.append(addr)
            continue
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        servers[addr] = srv
    if not servers:
        # Nothing to serve — exit loud so launchd retries and the log shows it.
        raise SystemExit(1)

    _log(
        f"serving {', '.join(f'{a}:{PORT}' for a in servers)} "
        f"-> engine {ENGINE[0]}:{ENGINE[1]}"
    )
    if pending:
        _log(f"will keep retrying: {', '.join(pending)}")

    def _retry_pending() -> None:
        """A configured address may appear after startup (VPN reconnecting)."""
        while True:
            time.sleep(30)
            for addr in list(pending):
                try:
                    srv = ThreadingHTTPServer((addr, PORT), ProxyHandler)
                except OSError:
                    continue
                pending.remove(addr)
                threading.Thread(target=srv.serve_forever, daemon=True).start()
                servers[addr] = srv
                _log(f"now serving {addr}:{PORT}")

    if pending:
        threading.Thread(target=_retry_pending, daemon=True).start()

    # Servers run in daemon threads; this thread keeps the process (and its
    # listeners) alive until a signal stops it.
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        _log("shutting down")
    finally:
        for srv in servers.values():
            srv.shutdown()


if __name__ == "__main__":
    main()
