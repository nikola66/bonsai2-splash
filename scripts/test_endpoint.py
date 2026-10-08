#!/usr/bin/env python3
"""Coding-agent point-of-view test of the local OpenAI-compatible endpoint.

Exercises: bind reachability, health, plain chat, tool calling (single /
parallel / multi-turn loop), streaming, JSON structured output, reasoning
efforts, long-context code tasks, and error handling. Logs every anomaly.

Includes regression checks for the defects fixed by openai_proxy.py:
loopback reachability, `content` never null on finish=length, and a missing
`model` field refused with 400.

stdlib only (urllib) so it runs anywhere.

    BASE defaults to http://127.0.0.1:8080/v1; override with BONSAI_TEST_BASE.
"""
import json
import os
import sys
import time
import urllib.request
import urllib.error

BASE = os.environ.get("BONSAI_TEST_BASE", "http://127.0.0.1:8080/v1").rstrip("/")
HOST = os.environ.get("BONSAI_TEST_HOST", "100.72.32.0")
MODEL = "bonsai-2-27b"

errors = []
notes = []
results = []


def log(section, ok, detail=""):
    line = f"[{'PASS' if ok else 'FAIL'}] {section}: {detail}"
    print(line, flush=True)
    results.append((section, ok, detail))
    if not ok:
        errors.append(line)


def post(path, payload, timeout=300):
    req = urllib.request.Request(
        BASE + path,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            body = r.read().decode()
            return r.status, json.loads(body), time.time() - t0, None
    except urllib.error.HTTPError as e:
        raw = e.read().decode(errors="replace")
        try:
            parsed = json.loads(raw)
        except Exception:
            parsed = raw
        return e.code, parsed, time.time() - t0, None
    except Exception as e:
        return None, None, time.time() - t0, str(e)


def chat(messages, **kw):
    payload = {"model": MODEL, "messages": messages, **kw}
    return post("/chat/completions", payload)


def first_content(resp):
    m = resp["choices"][0]["message"]
    return m.get("content") or "", m


def get_code(url, timeout=5):
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return r.status
    except urllib.error.HTTPError as e:
        return e.code
    except Exception:
        return None


# ------------------------------------------------------- 0. bind reachability
print("\n=== 0. Bind reachability (regression: loopback must be served) ===")
loop_code = get_code("http://127.0.0.1:8080/health")
log("bind: 127.0.0.1:8080/health", loop_code == 200, f"status={loop_code}")
bind_code = get_code(f"http://{HOST}:8080/health")
log(f"bind: {HOST}:8080/health", bind_code == 200, f"status={bind_code}")
engine_net = get_code("http://100.72.32.0:8081/health")
log("bind: engine port not network-reachable", engine_net is None,
    f"100.72.32.0:8081 -> {engine_net} (expect refused/None)")

# ---------------------------------------------------------------- 1. plain chat
print("\n=== 1. Plain chat (agent quick prompt) ===")
st, r, wall, err = chat(
    [{"role": "user", "content": "Reply with exactly: PONG"}],
    max_tokens=512, temperature=0,
)
if err or st != 200:
    log("plain chat", False, f"status={st} err={err} body={str(r)[:300]}")
else:
    content, msg = first_content(r)
    t = r.get("timings", {})
    log("plain chat", "PONG" in content,
        f"wall={wall:.2f}s content={content.strip()[:60]!r} finish={r['choices'][0]['finish_reason']} timings={json.dumps(t)[:300]}")
    if not t:
        notes.append("plain chat: no timings object in response")
    if "usage" not in r:
        notes.append("plain chat: no usage object in response")

# --------------------------------- 1b. content is never null (proxy fix)
st, r, wall, err = chat(
    [{"role": "user", "content": "Think hard: what is 2+2?"}],
    max_tokens=10, temperature=0,
)
if err or st != 200:
    log("content never null", False, f"status={st} err={err} body={str(r)[:300]}")
else:
    m = r["choices"][0]["message"]
    val = m.get("content")
    log("content never null (finish=length)",
        isinstance(val, str) and not m.get("tool_calls"),
        f"content={val!r} type={type(val).__name__} finish={r['choices'][0]['finish_reason']}")

# ------------------------------------------------- 2. single tool call (agent)
print("\n=== 2. Single tool call ===")
tools = [
    {
        "type": "function",
        "function": {
            "name": "grep_codebase",
            "description": "Search file contents by regex in the workspace",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "regex to find"},
                    "path": {"type": "string", "description": "directory to search"},
                },
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_shell",
            "description": "Run a shell command in the workspace",
            "parameters": {
                "type": "object",
                "properties": {"command": {"type": "string"}},
                "required": ["command"],
            },
        },
    },
]

st, r, wall, err = chat(
    [
        {"role": "system", "content": "You are a coding agent. Use tools when needed."},
        {"role": "user", "content": "Find where the server binds its port and what env var controls it."},
    ],
    tools=tools,
    max_tokens=512,
    temperature=0,
)
if err or st != 200:
    log("single tool call", False, f"status={st} err={err} body={str(r)[:400]}")
else:
    msg = r["choices"][0]["message"]
    fr = r["choices"][0]["finish_reason"]
    tc = msg.get("tool_calls") or []
    ok = fr == "tool_calls" and len(tc) >= 1
    detail = f"finish={fr} wall={wall:.2f}s n_tools={len(tc)}"
    if tc:
        names = [c["function"]["name"] for c in tc]
        detail += f" names={names}"
        for c in tc:
            try:
                json.loads(c["function"]["arguments"] or "{}")
            except Exception as e:
                ok = False
                detail += f" BAD_ARGS({c['function']['name']}): {e}"
            if not c.get("id"):
                ok = False
                detail += " MISSING_TOOL_CALL_ID"
        detail += f" args0={tc[0]['function']['arguments'][:120]!r}"
    log("single tool call", ok, detail)

# ------------------------------------------- 3. parallel tool calls
print("\n=== 3. Parallel tool calls ===")
st, r, wall, err = chat(
    [{"role": "user", "content": "Search the codebase for the bind address AND check server health in parallel."}],
    tools=tools + [{
        "type": "function",
        "function": {
            "name": "http_get",
            "description": "GET a URL and return the body",
            "parameters": {"type": "object", "properties": {"url": {"type": "string"}}, "required": ["url"]},
        },
    }],
    max_tokens=512,
    temperature=0,
)
if err or st != 200:
    log("parallel tool calls", False, f"status={st} err={err} body={str(r)[:300]}")
else:
    msg = r["choices"][0]["message"]
    tc = msg.get("tool_calls") or []
    log("parallel tool calls", len(tc) >= 2,
        f"finish={r['choices'][0]['finish_reason']} n_tools={len(tc)} names={[c['function']['name'] for c in tc]} wall={wall:.2f}s")

# ------------------------------------------- 4. multi-turn tool loop (agentic)
print("\n=== 4. Multi-turn tool loop (3 steps) ===")
loop_msgs = [
    {"role": "system", "content": "You are a coding agent. Use tools. Be terse."},
    {"role": "user", "content": "Which env var sets max context? Verify by running a command."},
]
loop_ok = True
loop_detail = []
for step in range(3):
    st, r, wall, err = chat(loop_msgs, tools=tools, max_tokens=512, temperature=0)
    if err or st != 200:
        loop_ok = False
        loop_detail.append(f"step{step}: HTTP {st} {err} {str(r)[:200]}")
        break
    msg = r["choices"][0]["message"]
    fr = r["choices"][0]["finish_reason"]
    tc = msg.get("tool_calls") or []
    loop_detail.append(f"step{step}: finish={fr} tools={[c['function']['name'] for c in tc]} wall={wall:.1f}s")
    if fr == "tool_calls" and tc:
        for c in tc:
            args = json.loads(c["function"]["arguments"] or "{}")
            fake = f'{{"stdout":"BONSAI_CTX=65536", "args":{json.dumps(args)}}}'
            loop_msgs.append({"role": "assistant", "content": None, "tool_calls": [c]})
            loop_msgs.append({"role": "tool", "tool_call_id": c["id"], "content": fake})
    else:
        break
log("multi-turn tool loop", loop_ok, " | ".join(loop_detail))

# ------------------------------------------- 5. streaming (SSE)
print("\n=== 5. Streaming ===")
req = urllib.request.Request(
    BASE + "/chat/completions",
    data=json.dumps({"model": MODEL, "messages": [{"role": "user", "content": "Count from 1 to 10, comma separated."}],
                     "max_tokens": 512, "temperature": 0, "stream": True}).encode(),
    headers={"Content-Type": "application/json"},
    method="POST",
)
t0 = time.time()
first_chunk_t = None
n_chunks = 0
finish = None
stream_text = []
try:
    with urllib.request.urlopen(req, timeout=300) as resp:
        if resp.headers.get_content_type() != "text/event-stream":
            log("streaming content-type", False, f"got {resp.headers.get_content_type()}")
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            try:
                obj = json.loads(data)
            except Exception:
                continue
            n_chunks += 1
            if first_chunk_t is None:
                first_chunk_t = time.time() - t0
            for ch in obj.get("choices", []):
                delta = ch.get("delta", {})
                if delta.get("content"):
                    stream_text.append(delta["content"])
                if ch.get("finish_reason"):
                    finish = ch["finish_reason"]
except Exception as e:
    log("streaming", False, f"exception: {e}")
else:
    total = time.time() - t0
    log("streaming", n_chunks > 0 and finish == "stop",
        f"chunks={n_chunks} ttft={first_chunk_t and round(first_chunk_t,2)}s total={total:.2f}s finish={finish} text={''.join(stream_text)[:80]!r}")

# ------------------------------------------- 6. streaming tool call deltas
print("\n=== 6. Streaming tool-call deltas ===")
req = urllib.request.Request(
    BASE + "/chat/completions",
    data=json.dumps({"model": MODEL, "messages": [{"role": "user", "content": "List files in src/."}],
                     "tools": tools, "max_tokens": 512, "temperature": 0, "stream": True}).encode(),
    headers={"Content-Type": "application/json"},
    method="POST",
)
n_chunks = 0
accum_args = ""
accum_name = ""
finish = None
got_tc_delta = False
try:
    with urllib.request.urlopen(req, timeout=300) as resp:
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            try:
                obj = json.loads(data)
            except Exception:
                continue
            n_chunks += 1
            for ch in obj.get("choices", []):
                delta = ch.get("delta", {})
                for part in delta.get("tool_calls") or []:
                    got_tc_delta = True
                    fn = part.get("function") or {}
                    accum_name += fn.get("name") or ""
                    accum_args += fn.get("arguments") or ""
                if ch.get("finish_reason"):
                    finish = ch["finish_reason"]
except Exception as e:
    log("streaming tool call", False, f"exception: {e}")
else:
    parse_ok = True
    parse_err = ""
    if accum_args:
        try:
            json.loads(accum_args)
        except Exception as e:
            parse_ok = False
            parse_err = str(e)
    log("streaming tool call", got_tc_delta and finish == "tool_calls" and parse_ok,
        f"chunks={n_chunks} finish={finish} name={accum_name!r} args={accum_args[:100]!r} parse_err={parse_err}")

# ------------------------------------------- 7. JSON structured output
print("\n=== 7. JSON structured output ===")
st, r, wall, err = chat(
    [{"role": "user", "content": 'Return a JSON object: {"file":"a.py","lines":10} . JSON only, no prose.'}],
    response_format={"type": "json_object"},
    max_tokens=512, temperature=0,
)
if err or st != 200:
    log("json response_format", False, f"status={st} err={err} body={str(r)[:300]}")
else:
    content, _ = first_content(r)
    try:
        parsed = json.loads(content)
        log("json response_format", True, f"ok={parsed} wall={wall:.2f}s")
    except Exception as e:
        log("json response_format", False, f"unparseable {e}: {content[:200]!r}")

# ------------------------------------------- 8. reasoning efforts
print("\n=== 8. Reasoning efforts ===")
for effort in ("none", "low", "high"):
    st, r, wall, err = chat(
        [{"role": "user", "content": "What is 17 * 23? Answer with just the number."}],
        reasoning_effort=effort, max_tokens=512, temperature=0,
    )
    if err or st != 200:
        log(f"reasoning_effort={effort}", False, f"status={st} err={err} body={str(r)[:250]}")
        continue
    content, msg = first_content(r)
    has_rc = "reasoning_content" in msg
    log(f"reasoning_effort={effort}", content.strip().startswith("391") or "391" in content,
        f"answer={content.strip()[:40]!r} wall={wall:.2f}s reasoning_content={'yes' if has_rc else 'no'}")

# ------------------------------------------- 9. code task w/ long context (search-like)
print("\n=== 9. Long-context code understanding (fake repo in prompt) ===")
files = []
for i in range(40):
    files.append(
        f"### src/module_{i}.py\n"
        + "\n".join(f"def func_{i}_{j}(x):\n    # line {j} of module {i}\n    return x * {j + i}  # placeholder {i}-{j}" for j in range(30))
    )
needle = "SECRET_TOKEN = 'bonsai-needle-7f3a'\n"
repo = "\n\n".join(files) + "\n\n### src/config.py\n" + needle + "DEBUG = False\n"
st, r, wall, err = chat(
    [{"role": "user", "content": f"Here is a repository:\n\n{repo}\n\nFind the secret token value and output only its value."}],
    max_tokens=300, temperature=0, reasoning_effort="none",
)
if err or st != 200:
    log("long-context needle", False, f"status={st} err={err} body={str(r)[:300]}")
else:
    content, _ = first_content(r)
    t = r.get("timings", {})
    log("long-context needle", "bonsai-needle-7f3a" in content,
        f"wall={wall:.1f}s answer={content.strip()[:60]!r} timings={json.dumps(t)[:250]}")

# ------------------------------------------- 10. error handling
print("\n=== 10. Error handling ===")
cases = [
    ("unknown model", {"model": "gpt-nonexistent", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 5}),
    ("missing messages", {"model": MODEL, "max_tokens": 5}),
    ("missing model", {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 5}),
    ("bad json body", None),
    ("tool result without tool_call_id", {"model": MODEL, "messages": [{"role": "tool", "content": "x"}], "max_tokens": 5}),
]
for name, payload in cases:
    if payload is None:
        req = urllib.request.Request(BASE + "/chat/completions", data=b"{not json",
                                     headers={"Content-Type": "application/json"}, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                st, body = resp.status, resp.read().decode()[:200]
        except urllib.error.HTTPError as e:
            st, body = e.code, e.read().decode()[:200]
        except Exception as e:
            st, body = None, str(e)
    else:
        st, body, _, _ = post("/chat/completions", payload, timeout=60)
    # `model` missing is the fixed spec deviation: it must be 400, not 200.
    if name == "missing model":
        good = st == 400
    else:
        good = st in (400, 404, 422)
    log(f"error: {name}", good, f"status={st} body={str(body)[:200]}")

# ------------------------------------------- 11. unsupported endpoint probing
print("\n=== 11. Agent-required endpoints ===")
for path, method in [("/embeddings", "POST"), ("/completions", "POST"), ("/chat/completions", "GET")]:
    req = urllib.request.Request(BASE + path, data=b"{}" if method == "POST" else None,
                                 headers={"Content-Type": "application/json"}, method=method)
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            st, body = resp.status, resp.read().decode()[:150]
    except urllib.error.HTTPError as e:
        st, body = e.code, e.read().decode()[:150]
    except Exception as e:
        st, body = None, str(e)
    log(f"{method} {path} reachable", st is not None, f"status={st} body={str(body)[:150]}")

# ------------------------------------------- summary
print("\n" + "=" * 70)
fails = [r for r in results if not r[1]]
print(f"TOTAL: {len(results)} checks, {len(fails)} failed")
for s, _, d in fails:
    print(f"  FAILED: {s}: {d}")
if notes:
    print("NOTES:")
    for n in notes:
        print(f"  {n}")
sys.exit(1 if fails else 0)
