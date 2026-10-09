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
| Context window | `65536` (raise with the chat-page slider or `BONSAI_CTX`) |
| Max output | `8192` is a safe client-side default |

`prism-ml/Ternary-Bonsai-2-27B-gguf:PQ2_0` also resolves to the same model if a
client insists on the full repo id.

The model supports **tool calling**, **image/PDF input**, and **reasoning**
(streamed separately as `reasoning_content`, controlled with
`reasoning_effort`). It does **not** serve embeddings.

If you exposed the server on a VPN/LAN address (see the README), the same URL
works with that host: `http://<your-address>:8080/v1`. Loopback always works too.

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
  context_length: 65536
  api_mode: chat_completions
```

- Hermes **rejects models advertising fewer than 64,000 tokens**, so it needs
  the 65536 default (or a larger `BONSAI_CTX`). If auto-detection from
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
            contextWindow: 65536,
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
| `maxInputTokens` | `65536` |
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
          "limit": { "context": 65536, "output": 8192 },
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

## Caveats that bite clients

| Symptom | Cause / fix |
|---|---|
| Truncated or empty answer, `finish_reason: length` | Reasoning shares the output budget. Send `reasoning_effort: "none"` for quick turns, or raise `max_tokens`. The server never returns `content: null` on a text reply (it normalizes to `""`). |
| `404` with a client base URL | Give the full base including `/v1` (e.g. `http://localhost:8080/v1`). The proxy also accepts the routes without `/v1` and with a trailing slash, but some clients add their own path segments. |
| `400 model is required` | The client omitted `model`; every request must name `bonsai-2-27b`. |
| Embeddings / RAG feature fails | The Splash engine serves no embeddings. `/v1/embeddings` returns a clear **501**; the Ollama bridge's `/api/embed*` returns 501. |
| Context looks too small | Default is 65536. Move the chat page's context slider (writes `.bonsai-ctx`) or set `BONSAI_CTX`; changing it restarts the server (~12–14 s). |
| Reasoning text appears in the answer | `reasoning_content` is a separate field; clients that ignore it show only the answer. To reduce it, lower `reasoning_effort` per request. |
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
