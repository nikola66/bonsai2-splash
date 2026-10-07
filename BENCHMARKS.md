# Benchmarks

Measurements from the maintainer's reference machine, produced by
`scripts/benchmark.py`. Every rate is read from the server's own `timings`
object — nothing here is estimated or copied from a vendor claim. Re-run the
script on your own hardware and compare; results will differ with memory
bandwidth, thermals and background load.

## Reference machine

| | |
|---|---|
| Mac | Mac mini, **Apple M6** |
| Unified memory | 24 GB |
| GPU cores | 12 |
| macOS | 27.0.1 |
| Low Power Mode | off (it throttles inference hard — check this before blaming the engine) |
| Engine | Splash 1.3.0 |
| Model | `prism-ml/Ternary-Bonsai-2-27B-gguf:PQ2_0` (6.75 GiB target weights) |
| Draft | `incoai/Qwen3.8-27B-DFlash2`, 1.18 GiB, always on |
| Context | 65536 (service default) |
| KV cache | `int8` |

## Results

Median of 3 repetitions, Splash serving on loopback, nothing else running.

| Measurement | Rate | Notes |
|---|---|---|
| **Prefill** | **293 tok/s** | ~1695-token synthetic prompt; range 290–295 |
| **Decode, reasoning off** | **68 tok/s** | TTFT 206 ms |
| **Decode, reasoning on** (`low`) | **50 tok/s** | ~106 reasoning tokens in the sample |
| **Vision prefill** | **253 tok/s** | 1024x768 image → 789 vision tokens |
| Weights → ready | **3.6 s** | from Splash's own startup log |
| Service start → serving | **~21 s** | `./bonsai.sh start`, weights already cached |

### Reproduce

```bash
python3 scripts/benchmark.py --base-url http://127.0.0.1:8080 \
    --reps 3 --max-tokens 400 --prefill-tokens 1600 --out reports
```

Writes a timestamped Markdown and JSON report to `reports/`. `--quick` runs a
single short repetition for a smoke check.

## Splash vs the previous llama.cpp fork

The same `PQ2_0` weights, the same machine, the same prompt — only the engine
changed. The old numbers come from the llama.cpp-based demo that preceded this
repo (same box, `PQ2_0` + MTP speculative decoding).

| Metric | llama.cpp fork + MTP | Splash | Change |
|---|---|---|---|
| Decode | 17–20 tok/s | **68 tok/s** | ~3.7x |
| Decode, reasoning on | ~17 tok/s | **50 tok/s** | ~2.9x |
| Prefill | ~200 tok/s | **293 tok/s** | ~1.4x |
| Weights → ready | ~14 GB RSS resident | 3.6 s, file-backed | — |
| Memory at rest | resident while serving | 18 GB free when stopped | ~11 GB returned |

The 3.7x decode gain comes from Splash's own Metal kernels plus the paired
DFlash2 speculative draft, which replaces the fork's MTP implementation. Splash
is also the reason the weights no longer sit fully resident in RAM.

Two behavioural differences worth knowing before you compare numbers:

- **Speculative decoding is always on** in Splash. There is no switch to turn it
  off; it is not separately configurable.
- **No embeddings endpoint.** `/v1/embeddings` returns 404 and the bridge's
  `/api/embed` returns 501. The old engine served them. Anything depending on
  embeddings will break.

## Methodology notes

Worth knowing if you write your own benchmark, because these two effects will
otherwise mislead you:

**Prefill must use a unique prompt per repetition.** Splash keeps a prefix cache.
On a cache hit, `timings.prompt_n` still counts the cached tokens while
`prompt_ms` covers only the *new* ones, so repeating one identical prompt reports
a rate several times too low. Measured on the reference machine: a repeated
10,728-token prompt reported **42 tok/s**; the same prompt with a randomised
prefix and a nonce reported **262–285 tok/s**. `scripts/benchmark.py` randomises
the prompt and records `cache_status` per sample so this cannot happen silently.

**Prefill rate scales inversely with prompt size on short prompts.** Under about
2000 tokens, a fixed setup cost dominates, so a short prompt reports a *lower*
rate than a long one (293 tok/s at ~1700 tokens vs 253 tok/s for a 789-token
image prompt, despite the latter being a different code path). Use prompts above
a few thousand tokens for a stable prefill number.

Decode numbers are less sensitive to this — the prompt is tiny and constant — but
the first repetition after a context change or an idle release is slower. The
script reports per-sample values in the JSON, so outliers are visible.

## Context and memory

Splash reports a memory plan per machine: on this 24 GB box it allows **188,409**
tokens of context, below the model's native 262,144. The plan is what fits
alongside the weights and KV cache within a recommended working set.

At `int8` KV (Splash's default) the cache costs about **32 KiB per token**:

| Context | KV cache |
|---|---|
| 65536 | ~2 GB |
| 131072 | ~4 GB |
| 262144 | ~8 GB |

Very long contexts on a tight machine can spill blocks to SSD with
`BONSAI_MAX_CACHE_DISK=16G` (real disk writes) or trade precision for memory
with `BONSAI_KV_FORMAT=bf16`, which doubles KV usage for a marginal quality gain.

Note that `ps` RSS understates the engine's real footprint — Splash memory-maps
the weights, so the process tree showed 3.3 GB RSS while the working set was
closer to 11 GB. Use `/status` for the authoritative plan, or compare free memory
before and after a stop (18.0 GB free stopped vs 5.0 GB running, measured here).