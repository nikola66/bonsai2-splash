# Connecting your apps

Bonsai serves a standard OpenAI-compatible API, so anything that can point at a
custom OpenAI base URL works with no glue code. This page gives copy-paste
configs for the clients the demo targets, plus the caveats worth knowing.

## What every client needs

| Setting | Value |
|---|---|
| Base URL | `http://localhost:8080/v1` |
| API key | **any non-empty string** — the server has no auth, e.g. `bonsai` |
| Model id | `bonsai-2-27b` |
| Context window | `131072` (the chat-page slider or `BONSAI_CTX` can change it) |
| Max output | `8192` — set this explicitly, see [Agent clients](#agent--tool-calling-clients) |

`prism-ml/Ternary-Bonsai-2-27B-gguf:PQ2_0` also resolves to the same model if a
client insists on the full repo id.

The model supports **tool calling**, **image/PDF input**, and **reasoning**
(streamed separately as `reasoning_content`, controlled with
`reasoning_effort`).

Embeddings are served too, by a separate EmbeddingGemma 2 model — send
`/v1/embeddings` requests with model `embeddinggemma-2-270m` (see
[Embeddings & RAG](#embeddings--rag) below).

If you exposed the server on a VPN/LAN address (see the README), the same URL
works with that host: `http://<your-address>:8080/v1`. Loopback always works too.

## Agent / tool-calling clients

Two things decide whether an agent loop is pleasant or abandoned, and both are
client-side settings rather than hardware.

**1. `max_tokens` must cover the thinking, or the turn comes back empty.**

`reasoning_effort` is spent from the same `max_tokens` budget the answer uses.
At `low` a single turn can spend 5,000 tokens thinking before writing anything,
so a client that caps output at 1,024 gets `content: ""` with
`finish_reason: "length"` — no text, no tool call, and a loop that stalls with
no error to react to. Measured on this box: `max_tokens: 1024` at `low` produced
an empty turn on 4 of 4 realistic agent tasks.

| Setting | Value |
|---|---|
| `max_tokens` | `8192` (**not** 512–1024) |
| `reasoning_effort` | `none` for tool-call turns |

Requests that carry a `tools` array and do **not** set `reasoning_effort` get
`none` automatically (`BONSAI_AGENT_EFFORT`, default `none`). Plain `/v1` calls
and the chat page keep the server default (`low`), so nothing changes for chat.

**2. Keep the prompt prefix stable, or pay a full re-prefill.**

The engine caches prefill by the longest common prefix of the *rendered*
prompt, and system messages plus tool definitions render at the very front of
it. Change anything there and every cached token behind it is discarded. On a
5.4K-token prompt:

| What changed | Time to first token |
|---|---|
| nothing (repeat) | 0.2 s |
| appended to the **end** of the system prompt | 0.5 s |
| timestamp **prepended** to the system prompt | **17.0 s** |
| tools **reordered** | **17.1 s** |
| one tool **appended** | **17.1 s** |

Real agent sessions at 28K tokens paid up to **115 s** for this. So:

- Put volatile content (clock, cwd, `git status`, todo/plan state, attached file
  list) at the **end** of the system prompt or the end of the message list —
  never the front.
- Keep the tool list fixed for the life of a session. The proxy sorts `tools` by
  function name before they reach the engine, so a client whose map iteration
  order varies is already safe; adding or removing a tool mid-session is not.
- The proxy logs an 8-character fingerprint of that prefix on every chat
  request (`prefix a1b2c3d4`). Two turns with the same fingerprint cannot have
  missed the cache; a changed fingerprint names the turn that broke it.

## Generic OpenAI SDK / curl

```bash
curl -s http://localhost:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"bonsai-2-27b","reasoning_effort":"none","max_tokens":200,
       "messages":[{"role":"user","content":"Write a Python quicksort"}]}'
```

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8080/v1", api_key="bonsai")
resp = client.chat.completions.create(
    model="bonsai-2-27b",
    messages=[{"role": "user", "content": "Write a Python quicksort"}],
    reasoning_effort="none",   # fast, no hidden thinking tokens
    max_tokens=512,
)
print(resp.choices[0].message.content)
```

Every response carries an extra top-level `timings` object (and a `metrics`
field on some routes); OpenAI SDKs ignore unknown fields, so it is safe to
leave in place — it is the authoritative performance data.

## Hermes Agent (Nous Research)

Run `hermes model`, choose **Custom endpoint (self-hosted / vLLM / etc.)**, and
enter `http://localhost:8080/v1`, any API key, and model `bonsai-2-27b`. Or put
it in `~/.hermes/config.yaml`:

```yaml
model:
  default: bonsai-2-27b
  provider: custom
  base_url: http://localhost:8080/v1
  api_key: bonsai          # any non-empty value
  context_length: 131072
  api_mode: chat_completions
```

- Hermes **rejects models advertising fewer than 64,000 tokens**; this server
  advertises 131072, so auto-detection works. If auto-detection from
  `/v1/models` is off, the explicit `context_length` above covers it.
- `api_mode: chat_completions` is the right mode; `anthropic_messages` and
  `codex_responses` are not applicable.
- For image input, point Hermes' auxiliary vision model at the same base URL,
  key and `bonsai-2-27b`.

## OpenClaw

Add a provider under `models.providers` (in `~/.openclaw/openclaw.json`, or via
`openclaw config`):

```json5
{
  agents: {
    defaults: { model: { primary: "bonsai/bonsai-2-27b" } },
  },
  models: {
    mode: "merge",
    providers: {
      bonsai: {
        baseUrl: "http://localhost:8080/v1",
        apiKey: "bonsai",
        api: "openai-completions",
        timeoutSeconds: 300,
        models: [
          {
            id: "bonsai-2-27b",
            name: "Bonsai 2 27B (local)",
            reasoning: true,
            input: ["text", "image"],
            contextWindow: 131072,
            maxTokens: 8192,
            cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
          },
        ],
      },
    },
  },
}
```

- `input: ["text", "image"]` is required if you want attachments passed to the
  model as images rather than text references.
- `reasoning: true` matches the model (it always emits reasoning content).
- `timeoutSeconds` is worth raising for local models and long agent runs.
- On a non-native `openai-completions` route OpenClaw automatically avoids
  OpenAI-only shaping (`service_tier`, `store`, prompt-cache hints) that a
  proxy would reject — no extra config needed.
- Run `openclaw doctor` to confirm the provider parses, then select
  `bonsai/bonsai-2-27b`.

## Visual Studio Code (Custom Endpoint, BYOK)

In the **Chat** view, open the language model picker → **Manage Language
Models** → **Add Models** → **Custom Endpoint**. Enter a group name, the model
display name, any API key, and choose **Chat Completions** as the API type.
VS Code writes `chatLanguageModels.json`; set these fields, then save and
restart VS Code:

| Field | Value |
|---|---|
| `id` | `bonsai-2-27b` |
| `name` | `Bonsai 2 27B` (shown in the picker) |
| `url` | `http://localhost:8080/v1/chat/completions` — the **full route**, not the base |
| `apiType` | `chat-completions` |
| `toolCalling` | `true` (agent mode needs this) |
| `vision` | `true` |
| `maxInputTokens` | `131072` |
| `maxOutputTokens` | `8192` |

Keep VS Code's generated `${apiKey}` input reference rather than pasting the key
into the file. The endpoint route matters: a base URL without
`/v1/chat/completions` is the most common cause of "route or API error".

## OpenCode

Add a custom provider (in `opencode.json`, or `~/.config/opencode/opencode.json`):

```json
{
  "$schema": "https://opencode.ai/config.json",
  "provider": {
    "bonsai": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "Bonsai (local)",
      "options": {
        "baseURL": "http://localhost:8080/v1",
        "apiKey": "bonsai"
      },
      "models": {
        "bonsai-2-27b": {
          "name": "Bonsai 2 27B",
          "limit": { "context": 131072, "output": 8192 },
          "tool_call": true,
          "reasoning": true
        }
      }
    }
  }
}
```

`@ai-sdk/openai-compatible` is bundled with OpenCode. Declaring `limit` matters:
without it OpenCode guesses the context budget and can send more than the server
accepts. Then select `bonsai/bonsai-2-27b`.

## LM Studio Bionic — not supported

Bionic is LM Studio's own agent app and runs models through **LM Studio's
runtime, LM Link, or Secure Cloud**. It has no "custom OpenAI base URL" provider,
so it cannot be pointed at the Bonsai server. Use the LM Studio **chat**
("Developer"/Playground) surface or any OpenAI-compatible client instead.

---

## Embeddings & RAG

RAG and semantic-search clients point at the **same base URL** and use the
embedding model id `embeddinggemma-2-270m`:

```bash
curl -s http://localhost:8080/v1/embeddings -H 'Content-Type: application/json' \
  -d '{"model":"embeddinggemma-2-270m","input":["Your document text goes here"],"dimensions":768}'
```

- Vectors are **768-dim**, L2-normalized. Ask for `dimensions` 128/256/512 and
  the service truncates (Matryoshka) and re-normalizes — useful when your index
  is fixed at a smaller size.
- `encoding_format: "base64"` is supported (float is the default).
- Ollama-based RAG works unchanged against the bridge
  (`model: "embeddinggemma-2-270m:latest"` on `/api/embed`).
- The embedding model is intentionally **not** listed in `/v1/models` or
  `/api/tags`: it exists for embed calls only, so chat clients can't pick it.
  Embed requests accept any model name; omitting it defaults to
  `embeddinggemma-2-270m`.

## Caveats that bite clients

| Symptom | Cause / fix |
|---|---|
| Empty answer, `finish_reason: length`, no tool call | Reasoning shares the output budget and ran out before the answer. Set `max_tokens: 8192`; agent turns already default to `reasoning_effort: none` when they carry `tools`. |
| `400 context_length_exceeded` | The conversation outgrew the window. The error now carries `limit` and `requested_tokens_estimate`, so the client knows how much to drop; the window can also be raised with the chat-page slider or `BONSAI_CTX` (restart, ~12–14 s). |
| One turn takes 30–120 s, the rest are fast | A prefix-cache miss, not a slow model. Something changed at the *front* of the prompt — see [Agent clients](#agent--tool-calling-clients). Compare the `prefix` fingerprints in the proxy log for the two turns. |
| Every turn is slower than the last | Expected: decode runs at 54 tok/s under 4K context and ~28 tok/s past 49K. Compaction, or fewer tools, is cheaper than more context. |
| Truncated or empty answer, `finish_reason: length` | Reasoning shares the output budget. Send `reasoning_effort: "none"` for quick turns, or raise `max_tokens`. The server never returns `content: null` on a text reply (it normalizes to `""`). |
| `404` with a client base URL | Give the full base including `/v1` (e.g. `http://localhost:8080/v1`). The proxy also accepts the routes without `/v1` and with a trailing slash, but some clients add their own path segments. |
| `400 model is required` | The client omitted `model`; every request must name `bonsai-2-27b`. |
| Embeddings / RAG feature fails | Shouldn't: `/v1/embeddings` works out of the box (EmbeddingGemma 2 via MLX). A **501** means embeddings are off (`BONSAI_EMBED=0`, or `mlx-vlm` missing from `.venv`); a **503** right after a restart means the ~580 MB model is still loading. |
| Context looks too small | Default is 131072. Move the chat page's context slider (writes `.bonsai-ctx`) or set `BONSAI_CTX`; changing it restarts the server (~12–14 s). |
| Reasoning text appears in the answer | `reasoning_content` is a separate field; clients that ignore it show only the answer. To reduce it, lower `reasoning_effort` per request. |
| Multi-turn chats get slower each turn | `preserve_thinking` is on by default: every turn's scratchpad stays in history, so prefill grows (measured: replaying 30 thinking tokens cost ~10% extra prefill on a tiny prompt — at `xhigh` with thousands of tokens it dominates). For long agent loops, send `preserve_thinking: false` or drop `reasoning_content` from replayed assistant messages. The proxy forwards the field untouched. |
| Images come back low-detail | The vision cap is 1024 tokens/image by default. Set `BONSAI_IMAGE_MAX_TOKENS=0` for full detail (4096). |

Browser apps on another origin are fine: the proxy answers CORS preflight and
reflects the origin, and it strips the incoming `Origin` before the engine sees
it (the engine rejects unknown origins by design).

## Verify your setup

```bash
curl -s http://localhost:8080/v1/models | python3 -m json.tool | head -20
python3 scripts/test_endpoint_clients.py     # client-compat regression suite
python3 scripts/test_endpoint.py             # full endpoint suite
```

Back to the [README](README.md).
