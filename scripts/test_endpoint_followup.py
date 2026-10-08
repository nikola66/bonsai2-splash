#!/usr/bin/env python3
"""Follow-up probes: diagnose failures + agent-specific behaviours.

BASE defaults to http://127.0.0.1:8080/v1; override with BONSAI_TEST_BASE.
"""
import json, os, time, urllib.request, urllib.error
from concurrent.futures import ThreadPoolExecutor

BASE = os.environ.get("BONSAI_TEST_BASE", "http://127.0.0.1:8080/v1").rstrip("/")
MODEL = "bonsai-2-27b"
results = []

def log(section, ok, detail=""):
    line = f"[{'PASS' if ok else 'FAIL'}] {section}: {detail}"
    print(line, flush=True)
    results.append((section, ok, detail))

def chat(messages, timeout=400, **kw):
    payload = {"model": MODEL, "messages": messages, **kw}
    req = urllib.request.Request(BASE + "/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read().decode()), time.time() - t0, None
    except urllib.error.HTTPError as e:
        try: body = json.loads(e.read().decode())
        except Exception: body = "unparseable"
        return e.code, body, time.time() - t0, None
    except Exception as e:
        return None, None, time.time() - t0, str(e)

print("=== A. Small max_tokens + default reasoning (defect: content null) ===")
st, r, w, err = chat([{"role": "user", "content": "Reply with exactly: PONG"}], max_tokens=20, temperature=0)
m = r["choices"][0]["message"]
# Reasoning eats the budget here, so content is empty — the fixed defect was
# that it was null (NoneType). It must now be a string.
log("A1 default effort, max_tokens=20: content is str, not None",
    isinstance(m.get("content"), str),
    f"content={m.get('content')!r} reasoning_len={len(m.get('reasoning_content') or '')} finish={r['choices'][0]['finish_reason']} "
    f"usage={json.dumps(r.get('usage'))[:200]}")

st, r, w, err = chat([{"role": "user", "content": "Reply with exactly: PONG"}],
                     max_tokens=20, temperature=0, reasoning_effort="none")
m = r["choices"][0]["message"]
log("A2 effort=none, max_tokens=20", (m.get("content") or "").strip() == "PONG",
    f"content={m.get('content')!r} finish={r['choices'][0]['finish_reason']} wall={w:.2f}s")

st, r, w, err = chat([{"role": "user", "content": "Reply with exactly: PONG"}],
                     max_tokens=512, temperature=0)
m = r["choices"][0]["message"]
log("A3 default effort, max_tokens=512", "PONG" in (m.get("content") or ""),
    f"content={m.get('content')!r} reasoning_len={len(m.get('reasoning_content') or '')} "
    f"finish={r['choices'][0]['finish_reason']} wall={w:.2f}s usage={json.dumps(r.get('usage'))[:200]}")

print("\n=== B. Streaming with default reasoning effort ===")
req = urllib.request.Request(BASE + "/chat/completions",
    data=json.dumps({"model": MODEL, "messages": [{"role": "user", "content": "Count 1 to 10, comma separated."}],
                     "max_tokens": 512, "temperature": 0, "stream": True}).encode(),
    headers={"Content-Type": "application/json"}, method="POST")
reason_chars, content_chars, finish = 0, 0, None
with urllib.request.urlopen(req, timeout=300) as resp:
    for raw in resp:
        line = raw.decode("utf-8", "replace").strip()
        if not line.startswith("data:"): continue
        d = line[5:].strip()
        if d == "[DONE]": break
        obj = json.loads(d)
        for ch in obj.get("choices", []):
            dl = ch.get("delta", {})
            if dl.get("reasoning_content"): reason_chars += len(dl["reasoning_content"])
            if dl.get("content"): content_chars += len(dl["content"])
            if ch.get("finish_reason"): finish = ch["finish_reason"]
log("B1 stream default effort, max_tokens=512", content_chars > 0 and finish == "stop",
    f"reasoning_chars={reason_chars} content_chars={content_chars} finish={finish}")

req = urllib.request.Request(BASE + "/chat/completions",
    data=json.dumps({"model": MODEL, "messages": [{"role": "user", "content": "Count 1 to 10, comma separated."}],
                     "max_tokens": 80, "temperature": 0, "stream": True,
                     "reasoning_effort": "none"}).encode(),
    headers={"Content-Type": "application/json"}, method="POST")
reason_chars, content_chars, finish = 0, 0, None
t0 = time.time(); ttft = None
with urllib.request.urlopen(req, timeout=300) as resp:
    for raw in resp:
        line = raw.decode("utf-8", "replace").strip()
        if not line.startswith("data:"): continue
        d = line[5:].strip()
        if d == "[DONE]": break
        obj = json.loads(d)
        for ch in obj.get("choices", []):
            dl = ch.get("delta", {})
            if dl.get("reasoning_content"): reason_chars += len(dl["reasoning_content"])
            if dl.get("content"):
                if ttft is None: ttft = time.time() - t0
                content_chars += len(dl["content"])
            if ch.get("finish_reason"): finish = ch["finish_reason"]
log("B2 stream effort=none, max_tokens=80", content_chars > 0 and finish == "stop",
    f"content_chars={content_chars} finish={finish} ttft={ttft and round(ttft,2)}s total={time.time()-t0:.2f}s")

print("\n=== C. Long-context needle (reasoning off) ===")
files = []
for i in range(40):
    files.append("### src/module_%d.py\n" % i + "\n".join(
        f"def func_{i}_{j}(x):\n    # line {j} of module {i}\n    return x * {j+i}  # placeholder {i}-{j}" for j in range(30)))
repo = "\n\n".join(files) + "\n\n### src/config.py\nSECRET_TOKEN = 'bonsai-needle-7f3a'\nDEBUG = False\n"
msgs = [{"role": "user", "content": f"Repository:\n\n{repo}\n\nFind the secret token value and output only its value."}]
st, r, w, err = chat(msgs, max_tokens=300, temperature=0, reasoning_effort="none")
m = r["choices"][0]["message"] if r else {}
content = (m.get("content") or "") if r else ""
log("C1 needle effort=none max_tokens=300", "bonsai-needle-7f3a" in content,
    f"wall={w:.1f}s content={content.strip()[:80]!r} timings={json.dumps((r or {}).get('timings', {}))[:220]}")

# cached re-run (prefix cache check): identical prompt, should be much faster prefill
st, r, w, err = chat(msgs, max_tokens=300, temperature=0, reasoning_effort="none")
t = (r or {}).get("timings", {})
log("C2 needle repeat (prefix cache)", t.get("prompt_ms", 9e9) < 60000,
    f"wall={w:.1f}s prompt_ms={t.get('prompt_ms')} prompt_per_second={round(t.get('prompt_per_second',0))} cache_n={t.get('cache_n')}")

print("\n=== D. Concurrency: agent firing parallel requests ===")
def one(i):
    t0 = time.time()
    st, r, w, err = chat([{"role": "user", "content": f"Reply with exactly: MSG{i}"}],
                         max_tokens=256, temperature=0, reasoning_effort="none", timeout=180)
    ok = r is not None and f"MSG{i}" in ((r["choices"][0]["message"].get("content") or ""))
    return (i, ok, w, err or (None if r else "no body"))
t0 = time.time()
with ThreadPoolExecutor(max_workers=4) as ex:
    out = list(ex.map(one, range(4)))
elapsed = time.time() - t0
allok = all(o[1] for o in out)
log("D1 4 parallel requests", allok,
    f"total={elapsed:.1f}s " + " ".join(f"#{i}:{'ok' if ok else 'BAD'} {w:.1f}s {e or ''}" for i, ok, w, e in out))

print("\n=== E. tool_choice forcing + strict schema ===")
tools = [{"type": "function", "function": {
    "name": "read_file", "description": "Read a file",
    "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}}]
st, r, w, err = chat([{"role": "user", "content": "read config.yaml"}],
                     tools=tools, tool_choice={"type": "function", "function": {"name": "read_file"}},
                     max_tokens=300, temperature=0)
tc = ((r or {}).get("choices", [{}])[0].get("message", {}).get("tool_calls")) if r else []
log("E1 forced tool_choice", bool(tc) and tc[0]["function"]["name"] == "read_file",
    f"status={st} tool_calls={[(c['function']['name'], c['function']['arguments']) for c in tc] if tc else None} err={err}")

print("\n=== F. OpenCode-style request extras tolerated ===")
st, r, w, err = chat([{"role": "user", "content": "Say OK"}],
                     max_tokens=256, temperature=0, top_p=0.95, n=1, stop=["END"],
                     user="opencode-test", seed=42, parallel_tool_calls=True,
                     reasoning_effort="none", logprobs=False)
log("F1 extra OpenAI fields", st == 200 and r is not None,
    f"status={st} err={err} body={str(r)[:150]}")

print("\n=== G. follow-up: HTTP keep-alive / sequential agent turn latency ===")
times = []
for i in range(3):
    st, r, w, err = chat([{"role": "user", "content": f"one-word answer: color {i}? (unique)"}],
                         max_tokens=256, temperature=0, reasoning_effort="none")
    times.append(w)
log("G1 sequential 3 turns", all(t < 10 for t in times), f"walls={[round(t,2) for t in times]}")

fails = [r for r in results if not r[1]]
print("\n" + "=" * 70)
print(f"TOTAL: {len(results)} checks, {len(fails)} failed")
for s, _, d in fails: print(f"  FAILED: {s}: {d}")
