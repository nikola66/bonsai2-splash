#!/usr/bin/env python3
"""Client-compatibility checks for the local OpenAI-compatible endpoint.

These cover the things real clients (Hermes, OpenClaw, VS Code, OpenCode and
generic OpenAI SDKs) rely on but the older suites did not exercise:

  * base URLs given with and without the /v1 suffix, and trailing slashes
  * CORS preflight + reflected headers for browser clients
    (and that a forwarded Origin no longer trips the engine's 403 guard)
  * /v1/models shape good enough to auto-detect context (>= 64000)
  * /v1/responses round-trip
  * max_completion_tokens honoured, json_schema, stream usage
  * the spec error envelope (`param` present)
  * a clear 501 for routes Splash does not serve

BASE defaults to http://127.0.0.1:8080/v1; override with BONSAI_TEST_BASE.
stdlib only (urllib), so it runs anywhere.
"""
import json
import os
import sys
import time
import urllib.error
import urllib.request

BASE = os.environ.get("BONSAI_TEST_BASE", "http://127.0.0.1:8080/v1").rstrip("/")
ROOT = BASE[: -len("/v1")] if BASE.endswith("/v1") else BASE
MODEL = "bonsai-2-27b"

results = []


def log(section, ok, detail=""):
    line = f"[{'PASS' if ok else 'FAIL'}] {section}: {detail}"
    print(line, flush=True)
    results.append((section, ok, detail))


def req(method, url, payload=None, headers=None, timeout=120):
    """-> (status, response_headers_lower, body_text)."""
    data = None if payload is None else json.dumps(payload).encode()
    hdrs = {"Content-Type": "application/json"}
    hdrs.update(headers or {})
    r = urllib.request.Request(url, data=data, headers=hdrs, method=method)
    t0 = time.time()
    try:
        with urllib.request.urlopen(r, timeout=timeout) as resp:
            return resp.status, {k.lower(): v for k, v in resp.headers.items()}, resp.read().decode(), time.time() - t0
    except urllib.error.HTTPError as e:
        return e.code, {k.lower(): v for k, v in e.headers.items()}, e.read().decode(errors="replace"), time.time() - t0
    except Exception as e:  # connection refused etc.
        return None, {}, str(e), time.time() - t0


def post(path, payload, headers=None, timeout=120):
    url = path if path.startswith("http") else BASE + path
    return req("POST", url, payload, headers, timeout)


def chat(messages, **kw):
    return post("/chat/completions", {"model": MODEL, "messages": messages, **kw})


print("=== A. Path tolerance (base URL with/without /v1, trailing slash) ===")
st, _, body, _ = post(ROOT + "/chat/completions",
                      {"model": MODEL, "reasoning_effort": "none", "max_tokens": 16,
                       "messages": [{"role": "user", "content": "Reply with exactly: OK"}]})
ok = st == 200 and "OK" in (json.loads(body)["choices"][0]["message"].get("content") or "") if st == 200 else False
log("alias /chat/completions (no /v1)", ok, f"status={st} body={body[:120]}")

st, _, body, _ = post(BASE + "/chat/completions/",
                      {"model": MODEL, "reasoning_effort": "none", "max_tokens": 16,
                       "messages": [{"role": "user", "content": "Reply with exactly: OK"}]})
log("trailing slash /v1/chat/completions/", st == 200, f"status={st} body={body[:80]}")

st, _, body, _ = req("GET", BASE + "/models/")
log("trailing slash /v1/models/", st == 200, f"status={st}")
st, _, body, _ = req("GET", ROOT + "/models")
log("alias /models (no /v1)", st == 200 and "data" in body, f"status={st} body={body[:80]}")

print("\n=== B. /v1/models is good enough to auto-detect context ===")
st, _, body, _ = req("GET", BASE + "/models")
model = None
if st == 200:
    data = json.loads(body)
    entries = data.get("data") if isinstance(data, dict) else None
    if isinstance(entries, list) and entries:
        model = entries[0]
    ok = (isinstance(data, dict) and data.get("object") == "list"
          and model is not None
          and (model.get("context_length") or model.get("max_model_len") or 0) >= 64000
          and bool(model.get("vision")))
    ctx = model.get("context_length") if model else None
else:
    ok, data = False, {}
log("models shape: object=list, context_length>=64000, vision", ok,
    f"status={st} id={model.get('id') if model else None} context_length={ctx}")

st, _, body, _ = req("GET", BASE + "/models/" + MODEL)
log("GET /v1/models/{id}", st == 200 and MODEL in body, f"status={st} body={body[:100]}")

print("\n=== C. CORS (browser clients) ===")
st, hdr, _, _ = req("OPTIONS", BASE + "/chat/completions",
                    headers={"Origin": "https://example.com",
                             "Access-Control-Request-Method": "POST",
                             "Access-Control-Request-Headers": "authorization,content-type"})
log("preflight 204 + Allow-Origin/Methods/Headers",
    st == 204 and hdr.get("access-control-allow-origin") == "https://example.com"
    and "POST" in hdr.get("access-control-allow-methods", "")
    and "authorization" in hdr.get("access-control-allow-headers", "").lower(),
    f"status={st} acao={hdr.get('access-control-allow-origin')} methods={hdr.get('access-control-allow-methods')}")

# The engine 403s on any Origin it was not started with; the proxy must strip it
# before forwarding and answer CORS itself.
st, hdr, body, _ = req("GET", BASE + "/models", headers={"Origin": "https://example.com"})
log("request with Origin not rejected by engine (403 guard)",
    st == 200 and hdr.get("access-control-allow-origin") == "https://example.com",
    f"status={st} acao={hdr.get('access-control-allow-origin')} body={body[:80]}")

st, hdr, _, _ = post("/chat/completions",
                     {"model": MODEL, "reasoning_effort": "none", "max_tokens": 16,
                      "messages": [{"role": "user", "content": "Say OK"}]},
                     headers={"Origin": "https://example.com"})
log("POST chat with Origin + reflected header",
    st == 200 and hdr.get("access-control-allow-origin") == "https://example.com",
    f"status={st} acao={hdr.get('access-control-allow-origin')}")

print("\n=== D. Error surfaces ===")
st, _, body, _ = post("/chat/completions", {"messages": [{"role": "user", "content": "hi"}]})
env = json.loads(body).get("error", {}) if st == 400 else {}
log("missing model -> 400 with OpenAI envelope (param present)",
    st == 400 and env.get("type") == "invalid_request_error" and "param" in env,
    f"status={st} error={json.dumps(env)[:140]}")

st, hdr, body, _ = req("GET", BASE + "/chat/completions")
log("GET on POST route -> 405 with Allow", st == 405 and "POST" in hdr.get("allow", ""),
    f"status={st} allow={hdr.get('allow')}")

st, _, body, _ = post("/embeddings", {"model": MODEL, "input": "hi"})
log("unsupported /v1/embeddings -> clear 501",
    st == 501 and "not served" in body.lower(), f"status={st} body={body[:140]}")

print("\n=== E. Modern OpenAI request shapes ===")
st, _, body, _ = post("/chat/completions",
                      {"model": MODEL, "reasoning_effort": "none", "max_completion_tokens": 6,
                       "messages": [{"role": "user", "content": "Count from 1 to 40, comma separated."}]})
usage = json.loads(body).get("usage", {}) if st == 200 else {}
log("max_completion_tokens honored", st == 200 and 0 < usage.get("completion_tokens", 99) <= 8,
    f"status={st} completion_tokens={usage.get('completion_tokens')} finish={json.loads(body)['choices'][0]['finish_reason'] if st == 200 else None}")

st, _, body, _ = post("/chat/completions",
                      {"model": MODEL, "reasoning_effort": "none", "max_tokens": 120,
                       "response_format": {"type": "json_schema", "json_schema": {
                           "name": "p", "schema": {"type": "object",
                                                   "properties": {"x": {"type": "integer"}},
                                                   "required": ["x"], "additionalProperties": False}}},
                       "messages": [{"role": "user", "content": "Return x=3 as JSON"}]})
parsed = None
if st == 200:
    try:
        parsed = json.loads(json.loads(body)["choices"][0]["message"].get("content") or "{}")
    except Exception:
        parsed = None
log("response_format json_schema", st == 200 and isinstance(parsed, dict) and "x" in parsed,
    f"status={st} content={parsed}")

st, _, body, _ = post("/chat/completions",
                      {"model": MODEL, "reasoning_effort": "none", "max_tokens": 32, "stream": True,
                       "stream_options": {"include_usage": True},
                       "messages": [{"role": "user", "content": "Say OK"}]})
usage = None
for line in body.splitlines():
    line = line.strip()
    if line.startswith("data:") and line[5:].strip() not in ("", "[DONE]"):
        obj = json.loads(line[5:].strip())
        if obj.get("usage"):
            usage = obj["usage"]
log("stream_options.include_usage (reasoning_tokens reported)",
    usage is not None and "reasoning_tokens" in json.dumps(usage.get("completion_tokens_details", {})),
    f"usage={json.dumps(usage)[:160]}")

st, _, body, _ = post("/responses", {"model": MODEL, "input": "say hi", "max_output_tokens": 120})
ok = st == 200 and '"output_text"' in body if st == 200 else False
log("/v1/responses round-trip", ok, f"status={st} body={body[:120]}")

print("\n" + "=" * 70)
fails = [r for r in results if not r[1]]
print(f"TOTAL: {len(results)} checks, {len(fails)} failed")
for s, _, d in fails:
    print(f"  FAILED: {s}: {d}")
sys.exit(1 if fails else 0)
