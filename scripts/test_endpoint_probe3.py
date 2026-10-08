#!/usr/bin/env python3
"""Final probes: OpenCode-specific request shapes + proxy regression checks.

BASE defaults to http://127.0.0.1:8080/v1; override with BONSAI_TEST_BASE.
"""
import json, os, time, urllib.request, urllib.error

BASE = os.environ.get("BONSAI_TEST_BASE", "http://127.0.0.1:8080/v1").rstrip("/")
HOST = os.environ.get("BONSAI_TEST_HOST", "100.72.32.0")
results = []
def log(s, ok, d=""):
    line = f"[{'PASS' if ok else 'FAIL'}] {s}: {d}"
    print(line, flush=True); results.append((s, ok, d))

def stream_usage(payload):
    req = urllib.request.Request(BASE + "/chat/completions", data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    usage = None; finish = None; content = []
    with urllib.request.urlopen(req, timeout=300) as resp:
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"): continue
            d = line[5:].strip()
            if d == "[DONE]": break
            obj = json.loads(d)
            if obj.get("usage"): usage = obj["usage"]
            for ch in obj.get("choices", []):
                if ch.get("delta", {}).get("content"): content.append(ch["delta"]["content"])
                if ch.get("finish_reason"): finish = ch["finish_reason"]
    return usage, finish, "".join(content)

print("=== H. OpenCode request shapes ===")
u, f, c = stream_usage({"model": "bonsai-2-27b", "messages": [{"role": "user", "content": "Say OK"}],
                         "max_tokens": 256, "reasoning_effort": "none", "stream": True,
                         "stream_options": {"include_usage": True}})
log("H1 stream_options.include_usage", u is not None and f == "stop",
    f"usage={json.dumps(u)} finish={f} content={c[:40]!r}")

u, f, c = stream_usage({"model": "bonsai-2-27b", "messages": [{"role": "user", "content": "Say OK"}],
                         "max_tokens": 256, "reasoning_effort": "none", "stream": True})
log("H2 stream without stream_options", f == "stop",
    f"usage={'present' if u else 'absent'} finish={f} content={c[:40]!r}")

for mid in ("bonsai-2-27b", "prism-ml/Ternary-Bonsai-2-27B-gguf:PQ2_0"):
    req = urllib.request.Request(BASE + "/chat/completions",
        data=json.dumps({"model": mid, "messages": [{"role": "user", "content": "Say OK"}],
                         "max_tokens": 256, "reasoning_effort": "none"}).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            obj = json.loads(r.read().decode())
            ok = "OK" in (obj["choices"][0]["message"].get("content") or "")
            log(f"H3 model id {mid[:40]}", ok, f"returned model={obj.get('model')!r}")
    except Exception as e:
        log(f"H3 model id {mid[:40]}", False, str(e))

# H4: content must be a string (never null) when finish=length without tool calls
req = urllib.request.Request(BASE + "/chat/completions",
    data=json.dumps({"model": "bonsai-2-27b", "messages": [{"role": "user", "content": "Think hard"}],
                     "max_tokens": 10}).encode(),
    headers={"Content-Type": "application/json"}, method="POST")
with urllib.request.urlopen(req, timeout=60) as r:
    obj = json.loads(r.read().decode())
    m = obj["choices"][0]["message"]
    val = m.get("content")
    log("H4 content is str (never null) when finish=length",
        isinstance(val, str) and not m.get("tool_calls"),
        f"content={val!r} type={type(val).__name__} "
        f"finish={obj['choices'][0]['finish_reason']} has_tool_calls={bool(m.get('tool_calls'))}")

print("\n=== I. Proxy regression checks ===")
# I1: loopback is served (the original blocker) and the bind address too.
def get_code(url, timeout=5):
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return r.status
    except urllib.error.HTTPError as e:
        return e.code
    except Exception:
        return None
log("I1 loopback 127.0.0.1:8080/health", get_code("http://127.0.0.1:8080/health") == 200,
    f"status={get_code('http://127.0.0.1:8080/health')}")
log(f"I1b bind address {HOST}:8080/health", get_code(f"http://{HOST}:8080/health") == 200,
    f"status={get_code(f'http://{HOST}:8080/health')}")

# I2: missing model must be a 400 invalid_request_error (spec), not 200.
req = urllib.request.Request(BASE + "/chat/completions",
    data=json.dumps({"messages": [{"role": "user", "content": "hi"}], "max_tokens": 10}).encode(),
    headers={"Content-Type": "application/json"}, method="POST")
try:
    with urllib.request.urlopen(req, timeout=60) as r:
        st, body = r.status, r.read().decode()[:200]
except urllib.error.HTTPError as e:
    st, body = e.code, e.read().decode()[:200]
except Exception as e:
    st, body = None, str(e)
err_shape = False
try:
    err_shape = json.loads(body)["error"]["type"] == "invalid_request_error"
except Exception:
    pass
log("I2 missing model -> 400 invalid_request_error", st == 400 and err_shape,
    f"status={st} body={body[:160]}")

# I3: the bundled chat page's request carries the injected model field.
req = urllib.request.Request(BASE.rsplit("/v1", 1)[0] + "/", method="GET")
try:
    with urllib.request.urlopen(req, timeout=60) as r:
        page = r.read().decode("utf-8", "replace")
    ok = 'JSON.stringify({model:"' in page and "JSON.stringify({messages:" not in page
    log("I3 chat page request has model injected", ok,
        "model field present" if ok else "pattern missing or not replaced")
except Exception as e:
    log("I3 chat page request has model injected", False, str(e))

fails = [r for r in results if not r[1]]
print(f"\nTOTAL: {len(results)} checks, {len(fails)} failed")
for s, _, d in fails: print(f"  FAILED: {s}: {d}")
