# Environment variables

Complete reference for this demo's user-configurable environment variables. The [README](README.md#configuration) covers the common ones; everything is listed here.

The demo runs on the **Splash** engine ([incoai/splash](https://github.com/incoai/splash)). `scripts/start_llama_server.sh` translates this demo's environment variables — and any legacy llama.cpp flags passed as extra arguments — into `splash serve` flags; Splash-native flags pass straight through (e.g. `--max-memory 20G`, `--kv-format bf16`, `--language-only`).

## Precedence

1. The real environment (what you export, or what `launchd` passes through).
2. `bonsai.env` in the repo root — copy `bonsai.env.example` and edit it. Read by `bonsai.sh`, `bonsai-service.sh`, `ollama_bridge.py` and `embeddings_service.py`, so the CLI and the service never disagree.
3. The defaults below.

`bonsai.env` is parsed as plain `KEY=value` pairs; it is not sourced as shell, so
a value cannot execute anything.

## Server (`scripts/start_llama_server.sh`, `bonsai-service.sh`)

| Variable | Default | Valid values | Purpose |
|----------|---------|--------------|---------|
| **Model** | | | |
| `BONSAI_FAMILY` | `bonsai2` | `bonsai2` | Model family. Splash serves the Bonsai 2 27B family only; any other value (e.g. the older `ternary` / `bonsai` families) is rejected with a clear error. |
| `BONSAI_MODEL` | `27B` | `27B` | Model size. Splash serves 27B only. |
| `BONSAI_SPLASH_MODEL` | `prism-ml/Ternary-Bonsai-2-27B-gguf:PQ2_0` | `OWNER/REPO[:VARIANT]` | The Hugging Face model Splash serves; it downloads the weights, paired DFlash2 draft and vision projector into its own cache on first run. Example alternative: `unsloth/Qwen3.8-27B-GGUF:UD-Q4_K_M`. |
| **Network** | | | |
| `BONSAI_HOST` | `127.0.0.1` | any bind address | Bind address of the public proxy (`openai_proxy.py`); `0.0.0.0` = all interfaces. **Loopback is always served too** — the proxy binds this address *and* `127.0.0.1`, and retries a missing address every 30 s while loopback serves. Also read by `bonsai.sh` (health checks) and by the bridge (its default `BRIDGE_HOST`). There is no auth on the server, so the bind address is the access control — keep it on loopback, or use a mesh-VPN address. |
| `PORT` | `8080` | port | Public HTTP port (the proxy's). |
| `BONSAI_ENGINE_PORT` | `PORT + 1` | port | Internal loopback port of the Splash engine; it never listens on the network. Override only if `PORT + 1` is already taken. Keep in sync across `start_llama_server.sh`, `bonsai.sh` and `ollama_bridge.py` (they read the same value). |
| `BRIDGE_PORT` | `11434` | port | Ollama bridge port. The official Ollama app binds `127.0.0.1:11434`, so there is no clash by default. |
| **Memory / context** | | | |
| `BONSAI_CTX` | auto (RAM-tiered 8K–128K) | `0`, or ≤ `262144` | Context window, passed as `--max-context`. `0`/unset = RAM-tiered default (**131072 on a 24 GB machine**, where the memory plan reserves KV for ~180K tokens at int8, so the cap costs no extra memory; 8192 / 16384 on smaller machines; never emitted as 0). An explicit number forces it (e.g. `262144` for the full training context). The launchd service reads `.bonsai-ctx` (written by the web UI's context slider) when `BONSAI_CTX` is unset, and otherwise falls back to the same tier. |
| `BONSAI_AGENT_EFFORT` | `none` | `none` … `max`, or `off` | `reasoning_effort` applied by the proxy to requests that carry a `tools` array and set none of their own. Thinking is spent from the same `max_tokens` the answer uses, so inheriting the server's thinking default made agent turns come back empty when a client capped output at ~1K. `off` disables the rewrite and leaves `--default-reasoning-effort` in charge. |
| `BONSAI_IDLE_RELEASE` | `off` | Splash duration (`30s`, `10m`, …) | Splash's idle release (`--idle-release`): after this idle time the engine frees model memory; the next request restores it from disk in seconds. Default `off` keeps the model resident — the old llama.cpp behavior; `10m` is a good choice on a 24 GB machine if you want the RAM back while idle. |
| `BONSAI_MAX_MEMORY` | Splash auto | Splash size (`28G`, `22G`) | Cap Splash's working-set budget (`--max-memory`); by default Splash derives it from free memory at start. |
| `BONSAI_MAX_CACHE_DISK` | unset | Splash size (`16G`) | SSD tier for KV/prefix cache blocks (`--max-cache-disk`). At very long context under memory pressure Splash spills blocks to disk (session temp files, i.e. real SSD writes). Only needed for very long contexts on tight machines. |
| `BONSAI_PERSISTENT_CACHE` | `0` | `1` | Keep the SSD cache across restarts (`--persistent-cache`; needs `BONSAI_MAX_CACHE_DISK`). Verified: a 6.4K prompt replayed in 0.6 s after a restart vs 20.2 s fresh (200 KV blocks restored, `taken_back.states: 1`). Each conversation ≥2048 tokens deep costs one restore point (~400 MB here, within the quota). Writes to SSD — leave off unless restarts (context-slider changes, upgrades) currently wipe your agent prefixes. |
| `BONSAI_KV_FORMAT` | `int8` (Splash default) | `int8`, `bf16` | KV cache format (`--kv-format`). `int8` uses ~half the KV memory of the old FP16 cache; `bf16` restores full precision (marginal quality gain at very long context, 2× KV memory). |
| **Vision / language** | | | |
| `BONSAI_IMAGE_MAX_TOKENS` | `1024` | number; `0` = uncapped | Cap on vision tokens per image, translated to Splash `--max-image-pixels` (tokens × 1024 px). `1024` is the fast consumer default (fine detail in large images lost); `0` passes Splash's own default of 4096 tokens / 4.2 MP (full detail — best for OCR/screenshots, slower on large images). |
| `BONSAI_LANGUAGE_ONLY` | `0` | `1` | Pass `--language-only`: skip the vision tower entirely (slightly lighter start; image/PDF input refused). |
| **Engine behavior** | | | |
| `BONSAI_OFFLINE` | `0` | `1` | Pass `--offline`: start the installed model without contacting the Hugging Face Hub. |
| `BONSAI_REASONING_EFFORT` | `medium` | `none`, `minimal`, `low`, `medium`, `high`, `xhigh`, `max` | Server default for requests that omit `reasoning_effort` (`--default-reasoning-effort`). A legacy `--reasoning-budget` CLI flag still wins over this. **Use `medium`** (the model card's recommended balance): upstream states `low` is *not supported* and "when selected the model will behave close to `xhigh`", so `low` silently buys near-maximum thinking rather than saving it. The model's own default is `xhigh`. Use per-request `reasoning_effort: "none"` for retrieval/reformat turns. |
| `HF_HOME` | `<checkout>/.cache/huggingface` | path | Where Splash's `huggingface_hub` keeps the weights, DFlash2 draft and vision projector (~12 GB), and where the MLX embeddings model lands. Defaulted inside the checkout so nothing is written to `~/.cache/huggingface`. |
| `BONSAI_CACHE_DIR` | `<checkout>/.cache/splash/prefix-cache` | path | Passed as `--cache-dir` with `--persistent-cache`, instead of Splash's default `~/Library/Caches/Splash/prefix-cache`. That default path is also symlinked here, so both names resolve to the same in-folder storage. |
| **Legacy — accepted but ignored (a note is printed)** | | | |
| `BONSAI_GGUF`, `BONSAI_MMPROJ` | — | path to a `.gguf` | Old llama.cpp local-file overrides. Splash loads by Hugging Face repo id only (`BONSAI_SPLASH_MODEL`). |
| `BONSAI_NGL`, `BONSAI_MMPROJ_CPU`, `BONSAI_SPECULATIVE`, `BONSAI_LLAMA_VERBOSE`, `BONSAI_KV4` | — | — | llama.cpp-only knobs. Ignored: Splash is Metal-only and always fully on-GPU, speculative decoding is always on (DFlash2), the KV cache is already `int8` by default, and the vision projector is managed by Splash. |

**Extra arguments** to `start_llama_server.sh` behave the same way:

- **Translated:** `-c` / `--ctx-size` → `--max-context`; `--alias` → `--served-model-name` + `--announce-served-name`; `--model`/`-m` → the served model id; `--image-max-tokens N` → `--max-image-pixels N*1024`; `--reasoning-budget` → `--default-reasoning-effort` (`0` → `none`, `≤512` → `low`, `≤2048` → `medium`, `≤8192` → `high`, else `max`); `--reasoning-budget -1` → engine model default. If no reasoning flag is supplied, the launcher defaults thinking to `BONSAI_REASONING_EFFORT` or `medium` (`--default-reasoning-effort`); `--chat-template-kwargs '{"enable_thinking": false}'` → `--default-reasoning-effort none`.
- **Ignored with a note:** `--jinja`, `-fa`, `-ngl`, `--temp/--top-p/--top-k/--min-p`, `--spec-type`, `--spec-draft-n-max`, `--parallel`/`-np`, `--mmproj`, `--cache-type-k/-v`, `--kv-mean-center`, `--cache-reuse`, `-ub`, `--reasoning-format`, `--webui-config-file`, `-v`, …
- **Passed through:** anything else is handed to `splash serve` verbatim (its own flags work here; a typo makes Splash print its usage).

## Embeddings (`embeddings_service.py`, optional)

Splash has no pooling head, so `/v1/embeddings` is served by a separate MLX
service (EmbeddingGemma 2, text-only 270M) started by the launcher as a third
child. It runs only when `mlx-vlm` is importable in `.venv`; otherwise the route
returns a clear 501. `BONSAI_EMBED=0` disables it explicitly.

| Variable | Default | Valid values | Purpose |
|----------|---------|--------------|---------|
| `BONSAI_EMBED` | `1` | `1`, `0` | Serve `/v1/embeddings` (via the MLX service). `0` makes the route answer 501. |
| `BONSAI_EMBED_MODEL` | `gahabeen/embeddinggemma-2-text-270m-mlx-bf16` | HF repo id | The text-only EmbeddingGemma 2 conversion mlx-vlm loads (first run downloads ~580 MB). |
| `BONSAI_EMBED_PORT` | `8082` | port | Loopback port of the embeddings service; only `openai_proxy.py` and the Ollama bridge talk to it. |
| `BONSAI_EMBED_DIM` | `768` | `1`–`768` | Default output size when a request omits `dimensions`. EmbeddingGemma is Matryoshka (MRL): smaller sizes (128/256/512) truncate and re-normalize. |
| `BONSAI_EMBED_MAX_TOKENS` | `8192` | number | Per-input token cap (EmbeddingGemma's native context is 8K tokens). |

`POST /v1/embeddings` accepts the OpenAI shape: `input` (string or list of
strings), optional `dimensions` (MRL truncation, re-normalized) and
`encoding_format` (`float`/`base64`). Responses carry the standard
`data[].embedding` + `usage` fields.

## Ollama bridge (`ollama_bridge.py`)

| Variable | Default | Purpose |
|---|---|---|
| `BRIDGE_HOST` | `BONSAI_HOST`, else `127.0.0.1` | Bridge bind address. Falls back to the server's bind address so both are reachable from the same place; loopback is always served as well (same rule as the proxy). No auth — keep it on loopback or behind an authenticating network. |
| `BRIDGE_PORT` | `11434` | Bind port. |
| `BRIDGE_UPSTREAM` | `http://<bind>:<PORT>` | The public server the bridge translates to (the proxy, not the engine directly). |
| `BRIDGE_OLLAMA_VERSION` | `0.5.13` | Version reported by `GET /api/version`. |
| `BRIDGE_KV_BYTES_PER_TOKEN` | `32768` | Display constant for the web UI's KV memory estimate (`int8` KV ≈ 32 KiB/token). |

## Splash itself

Auth, host/origin allowlists, request limits and the rest of Splash's server surface are upstream's `splash serve` options — see `splash serve --help` and [Splash's DEVELOPMENT.md](https://github.com/incoai/splash/blob/main/DEVELOPMENT.md). Anything not listed above can be passed as a trailing argument to `./scripts/start_llama_server.sh`.

For example, to require a real API key instead of relying on the bind address:

```bash
./scripts/start_llama_server.sh --api-key "$MY_KEY"   # or: SPLASH_API_KEY=...
```

## Benchmark (`scripts/benchmark.py`)

| Variable | Default | Purpose |
|---|---|---|
| `BENCHMARK_BASE_URL` | `http://127.0.0.1:8080` | Server to measure (also `--base-url`). |

Everything else is a command-line flag: `--model`, `--out`, `--reps`,
`--max-tokens`, `--prefill-tokens`, `--quick`.
