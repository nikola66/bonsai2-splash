# Agent guide — setting up the Bonsai demo

For AI agents (and humans) helping someone run this demo. Goal: pick the right
flags for the user's hardware and use case. The behaviour notes below were
measured on the maintainer's reference machine (M6, 24 GB, macOS 27.0.1) — every
API response carries a `timings` object, so measure on the user's own hardware
before promising performance.

Engine reference: [README.md](README.md). Variable reference:
[environment_variables.md](environment_variables.md). Client configs:
[CLIENTS.md](CLIENTS.md). Measured results: [BENCHMARKS.md](BENCHMARKS.md).

**This demo runs on the [Splash](https://github.com/incoai/splash) engine**
(`brew install incoai/tap/splash`), replacing an earlier custom llama.cpp fork.
Entry points, ports and environment variables are unchanged — the launcher
(`scripts/start_llama_server.sh`) translates them into `splash serve` flags.

## Why the 27B model

- **Vision** — image and PDF input end to end; the vision tower comes bundled
  with the model repo and Splash loads it automatically.
- **Agentic / tool calling** — native OpenAI `tool_calls`, verified with full
  tool round-trips (`finish_reason: "tool_calls"`).
- **Thinking** — a reasoning model; thoughts stream separately as
  `reasoning_content`, controlled per request with `reasoning_effort`.
- **Long context** — up to 262,144 tokens; a 24 GB machine's memory plan allows
  ~188K within its working-set budget at `int8` KV.
- **Tiny footprint** — the PQ2_0 band is 6.75 GB of weights; the whole running
  stack costs about 11 GB of memory pressure on a 24 GB Mac.

## The model

| `BONSAI_SPLASH_MODEL` | What it is |
|---|---|
| `prism-ml/Ternary-Bonsai-2-27B-gguf:PQ2_0` (**default**) | Bonsai 2 27B, ternary, PQ2_0 band. Splash downloads it plus the paired **DFlash2 draft** (`incoai/Qwen3.8-27B-DFlash2`) and the vision projector into its own cache on first run (~10 GB total). |
| any `OWNER/REPO[:VARIANT]` | Any GGUF repo Splash can identify, or an MLX repo (e.g. `unsloth/Qwen3.8-27B-GGUF:UD-Q4_K_M`). |

Only **Bonsai 2 27B** is supported. `BONSAI_FAMILY` / `BONSAI_MODEL` exist for
compatibility with older configs and are validated at startup: any pair other
than `bonsai2` / `27B` is rejected with a clear message. The older ternary and
1-bit families and the 8B/4B/1.7B sizes needed the retired llama.cpp fork.

Splash requires **M3+ and macOS 26.4+**. No CPU, Windows, Linux or MLX path
exists. Local `.gguf` files cannot be loaded — Splash identifies models from
Hugging Face repositories only.

## Knobs that matter

All extra args pass through the start script, e.g.
`./scripts/start_llama_server.sh --max-memory 20G --language-only`.

| Knob | What it does | Trade-off |
|---|---|---|
| `BONSAI_CTX=N` | Forces `--max-context N` (≤ 262144). Unset = RAM-tiered default (65536 on a 24 GB machine); the launchd service also reads `.bonsai-ctx` (chat-page slider, 8192–262144, default 65536). | More context = more KV memory. At `int8` KV ≈ 32 KiB/token, so 64K ≈ 2 GB and the machine's plan allows ~188K total. Pair very long contexts with `BONSAI_MAX_CACHE_DISK` on tight machines. |
| `BONSAI_IMAGE_MAX_TOKENS=N` | Vision tokens per image → `--max-image-pixels N*1024`. Default **1024**; `0` = uncapped (Splash's 4096 tokens / 4.2 MP). | **Ask the user** (below). |
| `BONSAI_IDLE_RELEASE=off\|30s\|10m…` | Splash frees model memory after N idle; the next request restores it from disk in seconds. Default `off` (stays resident). | On 24 GB, `10m` returns ~11 GB when idle but the first message after idle pays a reload. Good for machines that are off most of the day. |
| `BONSAI_MAX_MEMORY=size` | Caps Splash's working-set budget (`--max-memory 20G`). | Default: Splash derives it from free memory at start. |
| `BONSAI_MAX_CACHE_DISK=size` | SSD tier for KV/prefix cache blocks (`--max-cache-disk 16G`). | **Writes to disk.** Only for very long contexts on tight machines. |
| `BONSAI_KV_FORMAT` | `int8` (default, about half the KV memory of the old FP16) or `bf16`. | bf16 = full precision, 2× KV memory, marginal quality gain at very long context. |
| `BONSAI_LANGUAGE_ONLY=1` | Skip the vision tower entirely. | Slightly lighter and faster start; refuses image input. |
| `--reasoning-budget N` (legacy flag) | Mapped to `--default-reasoning-effort`: `0`→`none`, `≤512`→`low`, `≤2048`→`medium`, `≤8192`→`high`, else `max`. | Prefer `--default-reasoning-effort` or per-request `reasoning_effort`. |
| `--max-memory`, `--kv-format`, `--idle-release`, `--offline`, … | Splash-native flags passed straight through. | `splash serve --help` is the reference; unknown flags make Splash print usage. |

Legacy llama.cpp flags and env vars (`--jinja`, `-fa`, `-ngl`, `--parallel`,
`--temp/--top-p/…`, `BONSAI_GGUF`, `BONSAI_NGL`, `BONSAI_SPECULATIVE`, …) are
**accepted and ignored with one summary warning** so old configs keep working —
but they do nothing. Do not promise behaviour for them.

**Speculative decoding is always on**: Splash auto-pairs the Bonsai-trained
DFlash2 draft. The old `BONSAI_SPECULATIVE` switch is gone with the fork.

## Ask the user about these three

1. **Image detail** — large images are downscaled to 1024 vision tokens by
   default (fast; fine detail in big images and screenshots lost). Keep the cap,
   or `BONSAI_IMAGE_MAX_TOKENS=0` for full detail (best for OCR and small text,
   slower per large image)?
2. **Idle memory** — keep `BONSAI_IDLE_RELEASE=off` (always ready) or `10m`
   (reclaim ~11 GB while idle; first message after idle waits for a reload)?
3. **Context** — leave the service at 65536 / the slider value, or force
   `BONSAI_CTX`? Contexts over 100K cost real KV memory (32 KiB/token).

## What works with no extra setup

- **Chat page** at `/` — thinking picker, per-conversation sampling, image and
  PDF attachments. Settings live in the browser.
- **OpenAI-compatible API** at `/v1` — chat completions (streaming, tools),
  `/v1/models` (ids `bonsai-2-27b` and the full repo id), `/health`, `/ready`,
  `/status` (JSON: `maximum_context_tokens`, memory plan, KV info).
- **Thinking control** — per request `reasoning_effort`
  (`none`/`low`/`medium`/`high`/`max`); server default via
  `--default-reasoning-effort`, and the launcher supplies `low` whenever no
  reasoning flag is passed (the old `--reasoning-budget -1` opts back into the
  engine's model default). The old `thinking_budget_tokens` field is
  tolerated but ignored; unknown fields do not error.
- **MCP tools** are configured in the *client* (agent, IDE, app). The server
  serves a model, not MCP — there is nothing to seed server-side.
- **Ollama bridge** on port 11434: `/api/tags` (one entry, the
  `bonsai-2-27b:latest` alias; the duplicate repo id from `/v1/models` is hidden
  but both names resolve), `/api/chat` (NDJSON, `think` honored →
  `reasoning_effort`, images mapped to data URIs), `/api/version`, plus
  `GET/POST /bonsai/context` (writes `.bonsai-ctx` and restarts the server with
  the new `--max-context`; ~12–14 s, verified at 65536 → 131072 → 65536).

## Behaviour notes (measured: M6 24 GB, Splash 1.3.0)

- **Speed: 3–4x the old fork.** Code decode **68 tok/s** (old PQ2_0 + MTP:
  17–20); decode with reasoning on ~**50 tok/s**; prefill **~293 tok/s**;
  vision prefill ~253 tok/s. Read the `timings` object to verify on your hardware.
- **Startup is fast** — weights load in ~3.6 s; `bonsai.sh start` reports READY in
  ~15–21 s. The first-ever run downloads ~10 GB from Hugging Face first.
- **Thinking dominates "slow answers"**, not the engine. Cap it per chat with the
  reasoning-effort picker or `reasoning_effort` before blaming hardware.
- **Memory** — the running stack costs ~11 GB of pressure (engine RSS ~3.3 GB,
  since weights are file-backed); `./bonsai.sh stop` frees all of it (measured:
  18.0 GB free stopped vs 5.0 GB free running).
- **Process tree** — launcher (traps TERM/INT) → two children: `server.server`
  (the Splash engine; binds loopback `BONSAI_ENGINE_PORT`, default `PORT+1`;
  SIGTERM to it stops the whole tree, verified) → `serve-native` (holds the
  weights), and `openai_proxy.py` (owns the public `PORT`, binds `BONSAI_HOST`
  **and** `127.0.0.1` — loopback is always served). The proxy normalizes
  `content: null` → `""` (no `tool_calls`), enforces `model` with a 400, and
  injects the model id into the bundled chat page's request. The bridge's
  context restart SIGTERM-s `server.server` by its engine-port pattern (never
  the proxy); launchd's KeepAlive brings the pair back (~12–14 s) with the
  `.bonsai-ctx` value. A stray SIGTERM to the `splash serve` wrapper alone used
  to orphan the server — the launcher's trap prevents that.
- **Context changes restart the server** (weights reload). Expected, and the same
  as before.
- **No embeddings endpoint** — Splash's `/v1/embeddings` returns 404 and the
  bridge's `/api/embed` and `/api/embeddings` return 501 with
  `embeddings are not served by the Splash engine`. The old engine had them;
  anything relying on embeddings breaks.
- **Check macOS Low Power Mode** when speeds look far off (System Settings →
  Battery) — it throttles inference hard.
- **Prefix cache caveat** — when benchmarking, use a unique prompt per
  repetition; a repeated prompt reports prefill several times too low. Details in
  [BENCHMARKS.md](BENCHMARKS.md#methodology-notes).
- Windows, Linux and MLX paths are gone with the fork. Splash is Apple-only.

## Quick verification commands

```bash
./bonsai.sh status                  # agents, health, RSS, bind address

curl -s http://localhost:8080/health
curl -s http://localhost:8080/status | python3 -m json.tool | head -30
curl -s http://localhost:8080/v1/models | python3 -m json.tool

# time any request: read the "timings" object in the response
curl -s http://localhost:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"bonsai-2-27b","reasoning_effort":"none","max_tokens":200,
       "messages":[{"role":"user","content":"Write a Python quicksort"}]}' \
  | python3 -m json.tool | grep -A8 timings

# full report (writes reports/benchmark-<timestamp>.{md,json})
python3 scripts/benchmark.py --base-url http://localhost:8080

# endpoint regression suites (59 checks; BONSAI_TEST_BASE to point elsewhere)
python3 scripts/test_endpoint.py
python3 scripts/test_endpoint_followup.py
python3 scripts/test_endpoint_probe3.py
python3 scripts/test_endpoint_clients.py
```

Tool calling: send an OpenAI `tools` array and expect
`finish_reason: "tool_calls"` (verified). Vision: send an `image_url` content
part with a data URI (verified); image tokens respect the cap.