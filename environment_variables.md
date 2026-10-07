# Environment variables

Complete reference for this demo's user-configurable environment variables. The [README](README.md#configuration) covers the common ones; everything is listed here.

The demo runs on the **Splash** engine ([incoai/splash](https://github.com/incoai/splash)). `scripts/start_llama_server.sh` translates this demo's environment variables — and any legacy llama.cpp flags passed as extra arguments — into `splash serve` flags; Splash-native flags pass straight through (e.g. `--max-memory 20G`, `--kv-format bf16`, `--language-only`).

## Precedence

1. The real environment (what you export, or what `launchd` passes through).
2. `bonsai.env` in the repo root — copy `bonsai.env.example` and edit it. Read by `bonsai.sh`, `bonsai-service.sh` and `ollama_bridge.py`, so the CLI and the service never disagree.
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
| `BONSAI_HOST` | `127.0.0.1` | any bind address | Bind address (Splash `--host`); `0.0.0.0` = all interfaces. Also read by `bonsai.sh` (health checks) and by the bridge (its default `BRIDGE_HOST`). There is no auth on the server, so the bind address is the access control — keep it on loopback, or use a mesh-VPN address. |
| `PORT` | `8080` | port | HTTP port (Splash `--port`). |
| `BRIDGE_PORT` | `11434` | port | Ollama bridge port. The official Ollama app binds `127.0.0.1:11434`, so there is no clash by default. |
| **Memory / context** | | | |
| `BONSAI_CTX` | auto (RAM-tiered 8K–128K) | `0`, or ≤ `262144` | Context window, passed as `--max-context`. `0`/unset = RAM-tiered default (8192 → 131072 by machine memory; never emitted as 0). An explicit number forces it (e.g. `262144` for the full training context). The launchd service defaults to `65536` and additionally reads `.bonsai-ctx` (written by the web UI's context slider) when `BONSAI_CTX` is unset. |
| `BONSAI_IDLE_RELEASE` | `off` | Splash duration (`30s`, `10m`, …) | Splash's idle release (`--idle-release`): after this idle time the engine frees model memory; the next request restores it from disk in seconds. Default `off` keeps the model resident — the old llama.cpp behavior; `10m` is a good choice on a 24 GB machine if you want the RAM back while idle. |
| `BONSAI_MAX_MEMORY` | Splash auto | Splash size (`28G`, `22G`) | Cap Splash's working-set budget (`--max-memory`); by default Splash derives it from free memory at start. |
| `BONSAI_MAX_CACHE_DISK` | unset | Splash size (`16G`) | SSD tier for KV/prefix cache blocks (`--max-cache-disk`). At very long context under memory pressure Splash spills blocks to disk (session temp files, i.e. real SSD writes). Only needed for very long contexts on tight machines. |
| `BONSAI_KV_FORMAT` | `int8` (Splash default) | `int8`, `bf16` | KV cache format (`--kv-format`). `int8` uses ~half the KV memory of the old FP16 cache; `bf16` restores full precision (marginal quality gain at very long context, 2× KV memory). |
| **Vision / language** | | | |
| `BONSAI_IMAGE_MAX_TOKENS` | `1024` | number; `0` = uncapped | Cap on vision tokens per image, translated to Splash `--max-image-pixels` (tokens × 1024 px). `1024` is the fast consumer default (fine detail in large images lost); `0` passes Splash's own default of 4096 tokens / 4.2 MP (full detail — best for OCR/screenshots, slower on large images). |
| `BONSAI_LANGUAGE_ONLY` | `0` | `1` | Pass `--language-only`: skip the vision tower entirely (slightly lighter start; image/PDF input refused). |
| **Engine behavior** | | | |
| `BONSAI_OFFLINE` | `0` | `1` | Pass `--offline`: start the installed model without contacting the Hugging Face Hub. |
| **Legacy — accepted but ignored (a note is printed)** | | | |
| `BONSAI_GGUF`, `BONSAI_MMPROJ` | — | path to a `.gguf` | Old llama.cpp local-file overrides. Splash loads by Hugging Face repo id only (`BONSAI_SPLASH_MODEL`). |
| `BONSAI_NGL`, `BONSAI_MMPROJ_CPU`, `BONSAI_SPECULATIVE`, `BONSAI_LLAMA_VERBOSE`, `BONSAI_KV4` | — | — | llama.cpp-only knobs. Ignored: Splash is Metal-only and always fully on-GPU, speculative decoding is always on (DFlash2), the KV cache is already `int8` by default, and the vision projector is managed by Splash. |

**Extra arguments** to `start_llama_server.sh` behave the same way:

- **Translated:** `-c` / `--ctx-size` → `--max-context`; `--alias` → `--served-model-name` + `--announce-served-name`; `--model`/`-m` → the served model id; `--image-max-tokens N` → `--max-image-pixels N*1024`; `--reasoning-budget` → `--default-reasoning-effort` (`0` → `none`, `≤512` → `low`, `≤2048` → `medium`, `≤8192` → `high`, else `max`); `--chat-template-kwargs '{"enable_thinking": false}'` → `--default-reasoning-effort none`.
- **Ignored with a note:** `--jinja`, `-fa`, `-ngl`, `--temp/--top-p/--top-k/--min-p`, `--spec-type`, `--spec-draft-n-max`, `--parallel`/`-np`, `--mmproj`, `--cache-type-k/-v`, `--kv-mean-center`, `--cache-reuse`, `-ub`, `--reasoning-format`, `--webui-config-file`, `-v`, …
- **Passed through:** anything else is handed to `splash serve` verbatim (its own flags work here; a typo makes Splash print its usage).

## Ollama bridge (`ollama_bridge.py`)

| Variable | Default | Purpose |
|---|---|---|
| `BRIDGE_HOST` | `BONSAI_HOST`, else `127.0.0.1` | Bridge bind address. Falls back to the server's bind address so both are reachable from the same place. No auth — keep it on loopback or behind an authenticating network. |
| `BRIDGE_PORT` | `11434` | Bind port. |
| `BRIDGE_UPSTREAM` | `http://<bind>:<PORT>` | The Splash server the bridge translates to (also used to find the engine's PID for context restarts). |
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
