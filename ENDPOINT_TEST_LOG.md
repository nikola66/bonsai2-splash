# Endpoint test log — coding-agent (OpenCode) point of view
Tested: 2026-10-07 · Endpoint: `http://100.72.32.0:8080/v1` · Model: `bonsai-2-27b` (Splash, M6 24 GB)
Scripts: `agent_endpoint_test.py`, `agent_endpoint_followup.py`, `agent_endpoint_probe3.py` (30 checks total, 3 real defects found)

**Status: all 3 defects FIXED (2026-10-08) — see "Retest" at the bottom.**
Scripts moved to `scripts/test_endpoint.py`, `scripts/test_endpoint_followup.py`,
`scripts/test_endpoint_probe3.py` (43 checks total, `BONSAI_TEST_BASE` overridable,
default `http://127.0.0.1:8080/v1`). The fix lives in `openai_proxy.py`: a
normalization proxy that owns the public `PORT` while the Splash engine binds
loopback `PORT+1` (`BONSAI_ENGINE_PORT`).

## Errors to fix

### 1. BLOCKER — endpoint not reachable at `127.0.0.1:8080` — **FIXED**
- `connect to 127.0.0.1 port 8080 ... Connection refused`; `lsof` shows the server listening on
  **`100.72.32.0:8080` only** (Tailscale-style address). `./bonsai.sh status` confirms `bind: 100.72.32.0 present`.
- **Fix:** either bind `127.0.0.1` too (launchd plist / Splash `--host 127.0.0.1`, or add loopback),
  or use `http://100.72.32.0:8080/v1` in the OpenCode provider config. As given, the URL in the task does not work.
- **Resolved:** the engine (a single-address HTTP server) now binds `127.0.0.1:$BONSAI_ENGINE_PORT`
  (8081) only; `openai_proxy.py` owns `$PORT` (8080) and binds **both** `BONSAI_HOST` and
  `127.0.0.1` (with a 30 s rebind-retry when the VPN address is missing). The engine is no
  longer network-reachable at all. The Ollama bridge got the same treatment (loopback + `BONSAI_HOST`).

### 2. `content` is `null` (NoneType) when `finish_reason=length` — **FIXED**
- Default `reasoning_effort=low` shares the `max_tokens` budget with reasoning. With `max_tokens=10..20`
  the whole budget goes to reasoning → message is `{"content": null, "tool_calls": null}`, `finish=length`.
  Agent harnesses commonly do `message["content"].strip()` → **AttributeError / crash**.
- Evidence: `max_tokens=10` → `content=None type=NoneType`, `usage.completion_tokens_details.reasoning_tokens=10`.
- **Fix (server):** when there are no `tool_calls`, emit `content: ""` instead of `null`.
- **Mitigation (client):** keep `max_tokens` ≥ 256 on agent turns, or send `reasoning_effort: "none"`
  for quick tool-selection turns.
- **Resolved:** the proxy rewrites `content: null` → `""` whenever the message has no `tool_calls`
  (non-stream responses; streaming never emits null). Tool-call messages keep `content: null`,
  which is correct OpenAI shape.

### 3. Missing `model` field returns 200 instead of 400 — **FIXED**
- `{"messages": [...]}` without `model` → 200, served as `bonsai-2-27b`.
- OpenAI spec requires `model`. Low impact (OpenCode always sends it), but it hides client config bugs.
- **Fix:** reject with `400 model: field required`.
- **Resolved:** the proxy rejects a missing/empty/non-string `model` on `/v1/chat/completions`,
  `/v1/completions` and `/v1/responses` with
  `400 {"error": {"message": "model is required", "type": "invalid_request_error", ...}}`.
  The bundled chat page (which posts without `model`) gets the served id injected into its
  request body at serve time — if that pattern ever stops matching, enforcement backs off
  with a warning instead of breaking the UI.

## Behaviour notes (not bugs, but affect agent config)
- **Default reasoning is on** (`low`, set by the launcher): silently consumes `max_tokens`
  (e.g. `max_tokens=80` streaming produced 167 reasoning chars + 13 content chars, `finish=length`).
  Usage does report `reasoning_tokens` — good. Set `reasoning_effort` explicitly in OpenCode.
- **`/v1/embeddings` → 404, `/api/embed*` → 501** (engine limitation, documented): any embedding/RAG
  feature in a client will fail. No fix available on this engine.
- Long-context prefill is the dominant latency: 46K-token prompt = **172 s wall** (268 tok/s).
  Prefix cache makes the repeat **0.4 s** (`cache_n=46176`) — stable system prompts are effectively free
  on turn 2+.

## Verified working (agent-critical)
| Area | Result |
|---|---|
| Single / parallel / forced (`tool_choice`) tool calls | PASS — valid JSON args, ids present, `finish=tool_calls` |
| Multi-turn tool loop (3 steps: assistant→tool→assistant…) | PASS — 2.4–2.9 s/turn |
| Streaming (content + tool-call deltas) | PASS — args reassemble to valid JSON, correct `finish` |
| `stream_options: {include_usage: true}` | PASS — usage chunk present with `reasoning_tokens` |
| `response_format: json_object` | PASS |
| `reasoning_effort` none/low/high + `reasoning_content` | PASS — correct answers |
| Long-context needle retrieval (46K tokens) | PASS (with `reasoning_effort: none`) |
| 4 concurrent requests | PASS — all correct, 0.8 s total |
| Both model IDs (alias + full repo id) | PASS |
| Extra OpenAI fields (`seed`, `stop`, `user`, `top_p`, `parallel_tool_calls`, …) | PASS — tolerated |
| Error responses | PASS — proper OpenAI-style JSON 400/404 for bad JSON, unknown model, empty messages |
| `GET /v1/models`, `/health`, `/ready`, `/status` | PASS |

## Measured performance (M6, int8 KV, context 65536)
- Short reply, `reasoning_effort=none`: **0.20–0.29 s** wall; default effort: 0.5–0.6 s
- Decode: **~44–48 tok/s** (matches README's 50 tok/s with reasoning)
- Prefill: **~123–293 tok/s** short / **268 tok/s** at 46K
- TTFT (streaming, warm): **0.01–0.2 s**; sequential turns: 0.2 s

---

## Retest after fixes — 2026-10-08 · **43/43 PASS**

Service restarted on the new architecture:
launcher → `server.server` (loopback `127.0.0.1:8081`) + `openai_proxy.py`
(`127.0.0.1:8080` + `100.72.32.0:8080`); bridge on both addresses at 11434.

| Suite | Checks | Result |
|---|---|---|
| `scripts/test_endpoint.py` | 23 | 0 failed |
| `scripts/test_endpoint_followup.py` | 11 | 0 failed |
| `scripts/test_endpoint_probe3.py` | 9 | 0 failed |

Defect regressions now asserted on every run:
- **#1** `127.0.0.1:8080/health` → 200, bind address → 200, engine `100.72.32.0:8081` → refused.
- **#2** `max_tokens=10` → `content` is `str` (`''`), never `None`, `finish=length`, no `tool_calls`.
- **#3** missing `model` → exactly `400` with `invalid_request_error`; chat page carries the
  injected `model:"bonsai-2-27b"` field.

Also re-verified live: SSE streaming (33 chunks + `[DONE]`), tool calls
(`finish=tool_calls`, valid JSON args, `content:null` correctly preserved there),
prefix cache through the proxy (`cache_n=46176`, repeat 0.4 s), bridge
`/api/tags` + `/api/chat` on loopback **and** the VPN address, and a full
`/bonsai/context` restart cycle (SIGTERM matched the new engine-port pattern;
new tree came back with `--max-context=65536`, both addresses healthy).
