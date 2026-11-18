#!/usr/bin/env python3
"""Agent-loop regression checks for the local OpenAI-compatible endpoint.

These cover the two failure modes that made real coding agents stall on this
server -- both client-visible, neither a hardware limit:

  1. **Empty turns.** `reasoning_effort` is spent from the same `max_tokens` the
     answer uses, so a client that caps output near 1K got `content: ""` with
     `finish_reason: "length"`, no tool call, and no error to react to. The
     proxy now applies `BONSAI_AGENT_EFFORT` (default `none`) to tool-carrying
     requests that set no effort of their own.

  2. **Prefix-cache misses.** Splash caches prefill by the longest common
     prefix of the *rendered* prompt, and the tool list renders at the front.
     A client whose tool order varies therefore paid a full re-prefill (17 s on
     a 5.4K prompt, 115 s on a 28K session). The proxy sorts `tools` by
     function name, so the prefix depends on the tool set, not the order.

Also checks that a context refusal carries the limit and a size estimate (so a
harness can shrink instead of resending the same payload), and that a 20-turn
loop with a realistic tool count never produces an empty turn.

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
MODEL = os.environ.get("BONSAI_TEST_MODEL", "bonsai-2-27b")

results = []


def log(section, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {section}: {detail}", flush=True)
    results.append((section, ok, detail))


def req(method, url, payload=None, timeout=300):
    """-> (status, body_text, elapsed_seconds)."""
    data = None if payload is None else json.dumps(payload).encode()
    r = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}, method=method
    )
    t0 = time.time()
    try:
        with urllib.request.urlopen(r, timeout=timeout) as resp:
            return resp.status, resp.read().decode(errors="replace"), time.time() - t0
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode(errors="replace"), time.time() - t0
    except Exception as exc:  # connection refused etc.
        return None, str(exc), time.time() - t0


def post(path, payload, timeout=300):
    url = path if path.startswith("http") else BASE + path
    return req("POST", url, payload, timeout)


def get(path):
    url = path if path.startswith("http") else BASE + path
    return req("GET", url)


def tool(name, description="", props=None, required=None):
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description or f"{name} tool",
            "parameters": {
                "type": "object",
                "properties": props or {"path": {"type": "string"}},
                "required": required if required is not None else [],
            },
        },
    }


# A tool set the size of a real coding harness (OpenCode/Claude Code send 8-25).
AGENT_TOOLS = [
    tool(n)
    for n in (
        "read_file", "write_file", "edit", "apply_patch", "list_files", "glob",
        "grep", "run_command", "bash_output", "kill_shell", "git_status",
        "git_diff", "git_commit", "run_tests", "diagnostics", "todo_write",
        "plan_exit",
    )
]

AGENT_SYSTEM = (
    "You are a coding agent working in a repository. Keep diffs small, run the "
    "test suite after every change, and never modify tests to make them pass."
)


def chat(payload, timeout=300):
    return post("/chat/completions", payload, timeout)


def message_of(body):
    try:
        return json.loads(body)["choices"][0]
    except (ValueError, KeyError, IndexError):
        return {}


def empty_turn(choice):
    """An assistant turn that produced nothing the harness can act on."""
    msg = choice.get("message") or {}
    return not (msg.get("content") or "").strip() and not msg.get("tool_calls")


# ── 1. agent turns default to the configured effort ─────────────────────────


st, body, elapsed = chat(
    {
        "model": MODEL,
        "tools": AGENT_TOOLS,
        "max_tokens": 1024,
        "messages": [
            {"role": "system", "content": AGENT_SYSTEM},
            {"role": "user", "content": "Read src/app.py and summarise it."},
        ],
    }
)
choice = message_of(body)
reasoning = 0
try:
    reasoning = (
        json.loads(body)["usage"]["completion_tokens_details"]["reasoning_tokens"]
    )
except (ValueError, KeyError, TypeError):
    pass
log(
    "agent turn (tools, no effort, max_tokens=1024) is not empty",
    st == 200 and not empty_turn(choice),
    f"status={st} finish={choice.get('finish_reason')} "
    f"reasoning_tokens={reasoning} elapsed={elapsed:.1f}s",
)

st, body, _ = chat(
    {
        "model": MODEL,
        "tools": AGENT_TOOLS,
        "reasoning_effort": "low",
        "max_tokens": 1024,
        "messages": [
            {"role": "system", "content": AGENT_SYSTEM},
            {"role": "user", "content": "Read src/app.py and summarise it."},
        ],
    }
)
reasoning = 0
try:
    reasoning = (
        json.loads(body)["usage"]["completion_tokens_details"]["reasoning_tokens"]
    )
except (ValueError, KeyError, TypeError):
    pass
log(
    "explicit reasoning_effort is not overridden",
    st == 200 and reasoning > 0,
    f"status={st} reasoning_tokens={reasoning}",
)

# ── 2. tool order does not invalidate the prompt prefix ─────────────────────


def prompt_tokens(elapsed_body):
    try:
        return json.loads(elapsed_body)["timings"]
    except (ValueError, KeyError, TypeError):
        return {}


# A long, unique system prompt so a miss is expensive and attributable.
nonce = time.time_ns()
stable = (
    "Repository conventions follow.\n"
    + f"Session nonce {nonce}.\n"
    + "Rule: prefer small diffs and run the tests.\n" * 150
)
cold_payload = {
    "model": MODEL,
    "reasoning_effort": "none",
    "max_tokens": 8,
    "tools": AGENT_TOOLS,
    "messages": [
        {"role": "system", "content": stable},
        {"role": "user", "content": "hi"},
    ],
}
st, body, cold_secs = chat(cold_payload)
cold = prompt_tokens(body)
st2, body2, _ = chat({**cold_payload, "tools": list(reversed(AGENT_TOOLS))})
warm = prompt_tokens(body2)
hit = warm.get("cache_n") or 0
total = warm.get("prompt_n") or 0
ratio = (hit / total) if total else 0.0
log(
    "reordered tools still hit the prefix cache",
    st == 200 and st2 == 200 and ratio >= 0.9,
    f"cold={cold_secs:.1f}s cached={hit}/{total} ({ratio:.0%}) "
    f"warm_ttft={warm.get('prompt_ms', 0):.0f}ms",
)

# A prefix that is genuinely stable must hit too (guards against a proxy that
# rewrites the body per request, e.g. injecting a nonce).
st3, body3, _ = chat(cold_payload)
repeat = prompt_tokens(body3)
log(
    "identical request hits the prefix cache",
    st3 == 200 and (repeat.get("cache_n") or 0) > 0,
    f"cached={repeat.get('cache_n')}/{repeat.get('prompt_n')}",
)

# ── 3. context refusals are actionable ─────────────────────────────────────


st, body, _ = chat(
    {
        "model": MODEL,
        "reasoning_effort": "none",
        "max_tokens": 8,
        "messages": [{"role": "user", "content": "word " * 150000}],
    }
)
err = {}
try:
    err = json.loads(body).get("error") or {}
except ValueError:
    pass
log(
    "context refusal carries limit + estimate",
    st == 400
    and err.get("code") == "context_length_exceeded"
    and isinstance(err.get("limit"), int)
    and isinstance(err.get("requested_tokens_estimate"), int),
    f"status={st} code={err.get('code')} limit={err.get('limit')} "
    f"estimate={err.get('requested_tokens_estimate')}",
)

st, body, _ = get("/models")
advertised = None
try:
    for entry in json.loads(body).get("data") or []:
        if entry.get("id") == MODEL:
            advertised = entry.get("context_length") or entry.get("max_model_len")
            break
except ValueError:
    pass
st2, status_body, _ = get(ROOT + "/status")
served = None
try:
    served = json.loads(status_body).get("maximum_context_tokens")
except ValueError:
    pass
log(
    "/v1/models advertises the served window",
    isinstance(advertised, int)
    and isinstance(served, int)
    and advertised == served,
    f"models={advertised} status={served}",
)

# ── 4. a realistic agent loop never stalls ─────────────────────────────────

messages = [{"role": "system", "content": AGENT_SYSTEM}]
empty = 0
context_errors = 0
turns = 20
started = time.time()
for turn in range(turns):
    messages.append(
        {
            "role": "user",
            "content": f"Step {turn + 1}: inspect src/module_{turn}.py and report "
            f"the first function name.",
        }
    )
    st, body, _ = chat(
        {"model": MODEL, "tools": AGENT_TOOLS, "max_tokens": 1024, "messages": list(messages)}
    )
    if st == 400:
        context_errors += 1
        break
    choice = message_of(body)
    msg = choice.get("message") or {}
    if not msg:
        empty += 1
        break
    if empty_turn(choice):
        empty += 1
    calls = msg.get("tool_calls") or []
    messages.append(
        {
            "role": "assistant",
            "content": msg.get("content") or "",
            **({"tool_calls": calls} if calls else {}),
        }
    )
    if calls:
        for call in calls:
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call.get("id"),
                    "content": "def load_config(path, strict=False): ...",
                }
            )
    else:
        messages.append({"role": "user", "content": "Continue."})
loop_secs = time.time() - started

log(
    f"{turns}-turn agent loop with {len(AGENT_TOOLS)} tools has no empty turn",
    empty == 0 and context_errors == 0,
    f"empty={empty} context_errors={context_errors} "
    f"elapsed={loop_secs:.0f}s ({loop_secs / turns:.1f}s/turn)",
)

print("\n" + "=" * 70)
fails = [r for r in results if not r[1]]
print(f"TOTAL: {len(results)} checks, {len(fails)} failed")
for section, _, detail in fails:
    print(f"  FAILED: {section}: {detail}")
sys.exit(1 if fails else 0)
