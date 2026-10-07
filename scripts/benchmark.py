#!/usr/bin/env python3
"""Benchmark a running Bonsai server and emit a Markdown + JSON report.

Measures, against a live server (Splash or any OpenAI-compatible endpoint):
  1. startup / status facts (model id, context, KV format, memory plan)
  2. prefill throughput on a large synthetic prompt
  3. decode throughput, reasoning off
  4. decode throughput with reasoning on
  5. time-to-first-token on a short prompt
  6. vision prefill (one generated image) — skipped if the server has no vision
  7. tool-calling round trip (verifies `finish_reason: tool_calls`)
  8. idle RSS of the engine process tree, when it can be found locally

Usage:
    python3 scripts/benchmark.py                       # http://127.0.0.1:8080
    python3 scripts/benchmark.py --base-url http://host:8080 --out reports/
    python3 scripts/benchmark.py --quick               # fewer repetitions

No third-party dependencies: the standard library only, so it runs anywhere the
demo runs. Every number it prints comes from the server's own `timings` object,
so nothing here is estimated.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import random
import re
import secrets
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request
import zlib
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_BASE_URL = os.environ.get("BENCHMARK_BASE_URL", "http://127.0.0.1:8080")

# Prefill filler words. Prefill prompts are rebuilt with a fresh random order
# and a unique nonce for every repetition — see `unique_prefill_prompt` for why
# that matters: Splash caches prompt prefixes, so a repeated prompt would report
# only the uncached tail and make prefill look 6x slower than it is.
PREFILL_WORDS = (
    "alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima "
    "mike november oscar papa quebec romeo sierra tango uniform victor whiskey "
    "xray yankee zulu"
).split()
DECODE_PROMPT = "Write a Python function that merges two sorted lists and explain its complexity."
TOOL_PROMPT = "What is the weather in Oslo right now?"

TOOL_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a city.",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string", "description": "City name"}},
                "required": ["city"],
            },
        },
    }
]


# ── HTTP helpers (stdlib only) ────────────────────────────────────────────────
def http_json(
    url: str,
    payload: dict | None = None,
    timeout: float = 600.0,
    method: str | None = None,
) -> dict:
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(
        url,
        data=data,
        method=method or ("POST" if data is not None else "GET"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
        return json.loads(resp.read().decode() or "{}")


# ── A tiny generated PNG, so the vision test needs no fixture file ────────────
def make_test_png(width: int = 1024, height: int = 768) -> bytes:
    """Build a valid RGB PNG in memory: a grid plus high-contrast blocks.

    Deliberately simple geometry — the benchmark measures prefill throughput,
    not OCR accuracy.
    """
    rows = bytearray()
    for y in range(height):
        rows.append(0)  # PNG filter type: none
        for x in range(width):
            on_grid = x % 128 == 0 or y % 128 == 0
            if on_grid:
                px = (0, 0, 0)
            elif x < width // 2 and y < height // 2:
                px = (240, 240, 240)
            else:
                px = (30, 90, 180)
            rows.extend(px)

    def chunk(tag: bytes, data: bytes) -> bytes:
        return (
            len(data).to_bytes(4, "big")
            + tag
            + data
            + zlib.crc32(tag + data).to_bytes(4, "big")
        )

    ihdr = (
        width.to_bytes(4, "big")
        + height.to_bytes(4, "big")
        + bytes((8, 2, 0, 0, 0))  # 8-bit, truecolour RGB, no interlace
    )
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", ihdr)
        + chunk(b"IDAT", zlib.compress(bytes(rows), 6))
        + chunk(b"IEND", b"")
    )


# ── Individual measurements ───────────────────────────────────────────────────
def chat(base_url: str, model: str, messages: list, **fields) -> dict:
    body = {"model": model, "messages": messages}
    body.update(fields)
    started = time.monotonic()
    resp = http_json(f"{base_url}/v1/chat/completions", body)
    resp["_wall_seconds"] = time.monotonic() - started
    return resp


def timings_of(resp: dict) -> dict:
    return resp.get("timings") or {}


def result_text(resp: dict) -> str:
    choices = resp.get("choices") or [{}]
    return (choices[0].get("message") or {}).get("content") or ""


def unique_prefill_prompt(target_tokens: int) -> str:
    """A prompt of roughly `target_tokens` tokens that no prefix cache can match.

    Splash keeps a prefix cache, and on a hit its `timings.prompt_n` still counts
    the cached tokens while `prompt_ms` only covers the *new* ones. Repeating one
    identical prompt therefore reports a rate several times too low (measured:
    42 tok/s for a repeated 10.7K-token prompt vs 262 tok/s for the same prompt
    with a unique prefix). Randomising the order and prefixing a nonce forces a
    cache miss, so every repetition measures genuine prefill work.
    """
    rng = random.Random(secrets.token_hex(8))
    # One filler word is roughly 1.5 tokens; overshoot slightly and let the
    # server's own prompt_n be the authoritative token count.
    words = int(target_tokens / 1.5) + 16
    body = " ".join(rng.choice(PREFILL_WORDS) for _ in range(words))
    return f"nonce {secrets.token_hex(12)}\n{body}"


def bench_prefill(base_url: str, model: str, target_tokens: int, reps: int) -> dict:
    """Time prefill on a fresh prompt per repetition (see `unique_prefill_prompt`)."""
    samples = []
    for _ in range(reps):
        resp = chat(
            base_url,
            model,
            [{"role": "user", "content": unique_prefill_prompt(target_tokens)}],
            reasoning_effort="none",
            max_tokens=8,
        )
        t = timings_of(resp)
        cache = (resp.get("metrics") or {}).get("cache") or {}
        samples.append(
            {
                "prompt_tokens": t.get("prompt_n"),
                "cache_status": cache.get("status"),
                "cached_tokens": t.get("cache_n"),
                "prompt_ms": t.get("prompt_ms"),
                "prompt_per_second": t.get("prompt_per_second"),
            }
        )
    return {
        "target_tokens": target_tokens,
        "samples": samples,
        "median_prompt_tokens": _median([s["prompt_tokens"] for s in samples]),
        "median_prompt_per_second": _median([s["prompt_per_second"] for s in samples]),
        "min_prompt_per_second": min(
            (s["prompt_per_second"] for s in samples if s["prompt_per_second"]), default=None
        ),
        "max_prompt_per_second": max(
            (s["prompt_per_second"] for s in samples if s["prompt_per_second"]), default=None
        ),
    }


def bench_decode(
    base_url: str,
    model: str,
    max_tokens: int,
    reps: int,
    reasoning_effort: str | None,
    prompt: str = DECODE_PROMPT,
) -> dict:
    samples = []
    reasoning_tokens = []
    for _ in range(reps):
        fields = {"max_tokens": max_tokens}
        if reasoning_effort is not None:
            fields["reasoning_effort"] = reasoning_effort
        resp = chat(base_url, model, [{"role": "user", "content": prompt}], **fields)
        t = timings_of(resp)
        reasoning_tokens.append(
            (resp.get("usage") or {}).get("completion_tokens_details", {}).get(
                "reasoning_tokens", 0
            )
        )
        samples.append(
            {
                "predicted_tokens": t.get("predicted_n"),
                "predicted_per_second": t.get("predicted_per_second"),
                "ttft_ms": t.get("prompt_ms"),
                "cache_status": ((resp.get("metrics") or {}).get("cache") or {}).get("status"),
            }
        )
    return {
        "reasoning_effort": reasoning_effort or "(model default)",
        "prompt": prompt,
        "samples": samples,
        "median_decode_tokens_per_second": _median(
            [s["predicted_per_second"] for s in samples]
        ),
        "median_ttft_ms": _median([s["ttft_ms"] for s in samples]),
        "median_reasoning_tokens": _median(reasoning_tokens),
    }


def bench_vision(base_url: str, model: str) -> dict:
    import base64 as b64

    png = b64.b64encode(make_test_png()).decode()
    started = time.monotonic()
    resp = chat(
        base_url,
        model,
        [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this image in one sentence."},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{png}"},
                    },
                ],
            }
        ],
        reasoning_effort="none",
        max_tokens=64,
    )
    wall = time.monotonic() - started
    if "error" in resp:
        return {"supported": False, "error": str(resp["error"])[:300]}
    t = timings_of(resp)
    return {
        "supported": True,
        "image": "1024x768 generated PNG (no fixture file needed)",
        "prompt_tokens": t.get("prompt_n"),
        "cache_status": ((resp.get("metrics") or {}).get("cache") or {}).get("status"),
        "prompt_ms": t.get("prompt_ms"),
        "prompt_per_second": t.get("prompt_per_second"),
        "predicted_per_second": t.get("predicted_per_second"),
        "wall_seconds": round(wall, 2),
        "reply_sample": result_text(resp).strip().replace("\n", " ")[:160],
    }


def bench_tools(base_url: str, model: str) -> dict:
    resp = chat(
        base_url,
        model,
        [{"role": "user", "content": TOOL_PROMPT}],
        tools=TOOL_SCHEMA,
        reasoning_effort="none",
        max_tokens=256,
    )
    if "error" in resp:
        return {"supported": False, "error": str(resp["error"])[:300]}
    choice = (resp.get("choices") or [{}])[0]
    calls = (choice.get("message") or {}).get("tool_calls") or []
    return {
        "supported": bool(calls),
        "finish_reason": choice.get("finish_reason"),
        "tool_calls": [
            {"name": c.get("function", {}).get("name"), "arguments": c.get("function", {}).get("arguments")}
            for c in calls
        ],
    }


def engine_memory() -> dict:
    """RSS of the engine process tree, if the engine runs on this machine."""
    if platform.system() != "Darwin":
        return {"available": False, "reason": "local process scan is macOS-only"}
    pattern = re.compile(r"server\.server .*--port[= ]\d+|serve-native|splash serve")
    try:
        out = subprocess.run(
            ["ps", "-Ao", "pid=,ppid=,rss=,command="],
            capture_output=True,
            text=True,
            timeout=20,
            check=True,
        ).stdout
    except (OSError, subprocess.SubprocessError) as exc:
        return {"available": False, "reason": f"ps failed: {exc}"}

    rows = [ln.split(None, 3) for ln in out.splitlines() if len(ln.split(None, 3)) == 4]
    pids = {int(r[0]) for r in rows if pattern.search(r[3])}
    # Include direct children (the engine process holding the weights).
    for _ in range(2):
        pids |= {int(r[0]) for r in rows if int(r[1]) in pids}
    if not pids:
        return {"available": False, "reason": "engine process not found on this machine"}
    total_kb = sum(int(r[2]) for r in rows if int(r[0]) in pids)
    return {
        "available": True,
        "processes": sorted(pids),
        "total_rss_bytes": total_kb * 1024,
        "total_rss_gib": round(total_kb * 1024 / 1024**3, 2),
    }


def free_memory_bytes() -> int | None:
    if platform.system() == "Darwin":
        try:
            out = subprocess.run(
                ["vm_stat"], capture_output=True, text=True, timeout=15, check=True
            ).stdout
        except (OSError, subprocess.SubprocessError):
            return None
        page_size = 4096
        m = re.search(r"page size of (\d+) bytes", out)
        if m:
            page_size = int(m.group(1))
        free = inactive = speculative = 0
        for line in out.splitlines():
            if line.startswith("Pages free"):
                free = int(line.split(":")[1].strip().rstrip("."))
            elif line.startswith("Pages inactive"):
                inactive = int(line.split(":")[1].strip().rstrip("."))
            elif line.startswith("Pages speculative"):
                speculative = int(line.split(":")[1].strip().rstrip("."))
        return (free + inactive + speculative) * page_size
    try:
        with open("/proc/meminfo") as fh:  # pragma: no cover - non-macOS
            for line in fh:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    return None


def _median(values: list) -> float | None:
    nums = [v for v in values if isinstance(v, (int, float))]
    if not nums:
        return None
    return round(statistics.median(nums), 2)


def _num(value) -> str:
    """Format a possibly-absent number for a Markdown table cell."""
    if not isinstance(value, (int, float)):
        return "n/a"
    return f"{value:.1f}".rstrip("0").rstrip(".") if value % 1 else f"{int(value)}"


def round_floats(obj, digits: int = 2):
    if isinstance(obj, float):
        return round(obj, digits)
    if isinstance(obj, dict):
        return {k: round_floats(v, digits) for k, v in obj.items()}
    if isinstance(obj, list):
        return [round_floats(v, digits) for v in obj]
    return obj


# ── Report rendering ──────────────────────────────────────────────────────────
def hardware_facts() -> dict:
    facts = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "macos_version": None,
        "cpu": None,
        "gpu_cores": None,
        "memory_bytes": None,
        "low_power_mode": None,
    }
    if platform.system() == "Darwin":
        try:
            facts["macos_version"] = subprocess.run(
                ["sw_vers", "-productVersion"], capture_output=True, text=True, timeout=10
            ).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            pass
        for key, cmd in (
            ("cpu", ["sysctl", "-n", "machdep.cpu.brand_string"]),
            ("memory_bytes", ["sysctl", "-n", "hw.memsize"]),
            # hw.perflevel0 is the efficiency cluster; hw.ncpu is the total.
            ("cpu_cores", ["sysctl", "-n", "hw.ncpu"]),
        ):
            try:
                value = subprocess.run(
                    cmd, capture_output=True, text=True, timeout=10
                ).stdout.strip()
                if not value:
                    continue
                facts[key] = (
                    int(value)
                    if key in ("memory_bytes", "cpu_cores") and value.isdigit()
                    else value
                )
            except (OSError, subprocess.SubprocessError):
                pass
        try:
            pmset = subprocess.run(
                ["pmset", "-g"], capture_output=True, text=True, timeout=10
            ).stdout
            facts["low_power_mode"] = "on" if "lowpowermode 1" in pmset else "off"
        except (OSError, subprocess.SubprocessError):
            pass
    free = free_memory_bytes()
    if free:
        facts["available_memory_gib"] = round(free / 1024**3, 2)
    return {k: v for k, v in facts.items() if v is not None}


def render_markdown(report: dict) -> str:
    s = report["server"]
    hw = report["hardware"]
    m = report["measurements"]
    lines: list[str] = []
    a = lines.append

    a("# Benchmark report")
    a("")
    a(f"Generated {report['generated_at']} by `scripts/benchmark.py`.")
    a("Every rate comes from the server's own `timings` object.")
    a("")
    a("## Machine under test")
    a("")
    a("| Fact | Value |")
    a("|---|---|")
    for key, label in (
        ("cpu", "CPU"),
        ("cpu_cores", "CPU cores"),
        ("gpu_core_count", "GPU cores (engine-reported)"),
        ("memory_gib", "Unified memory (GiB)"),
        ("available_memory_gib", "Free memory at run time (GiB)"),
        ("macos_version", "macOS"),
        ("platform", "Platform"),
    ):
        value = s.get(key) or hw.get(key)
        if value is not None:
            a(f"| {label} | {value} |")
    if hw.get("low_power_mode"):
        a(f"| macOS Low Power Mode | {hw['low_power_mode']} |")
    a("")

    a("## Server under test")
    a("")
    a("| Fact | Value |")
    a("|---|---|")
    for key, label in (
        ("base_url", "Base URL"),
        ("model", "Model id (served)"),
        ("upstream_model", "Upstream model repo"),
        ("engine", "Engine"),
        ("max_context", "Context window"),
        ("kv_format", "KV format"),
        ("native_max_context", "Model's native context limit"),
        ("memory_plan_max_context", "Max context within this machine's memory plan"),
        ("target_weights_gib", "Target weights (GiB)"),
        ("draft_weights_gib", "DFlash2 draft weights (GiB)"),
        ("memory_pressure", "Memory pressure at run time"),
    ):
        if s.get(key) is not None:
            a(f"| {label} | `{s[key]}` |")
    a("")

    a("## Throughput")
    a("")
    a("| Measurement | Median | Notes |")
    a("|---|---|---|")
    pf = m["prefill"]
    spread = ""
    if pf.get("min_prompt_per_second") is not None and pf.get("max_prompt_per_second"):
        spread = (
            f"; range {pf['min_prompt_per_second']:.0f}–{pf['max_prompt_per_second']:.0f}"
        )
    a(
        f"| Prefill | {_num(pf.get('median_prompt_per_second'))} tok/s | "
        f"{pf.get('median_prompt_tokens')}-token synthetic prompt"
        f"{spread} (unique prompt per rep, so no prefix-cache hit) |"
    )
    d_off = m["decode_no_reasoning"]
    a(
        f"| Decode (reasoning off) | {_num(d_off.get('median_decode_tokens_per_second'))} tok/s | "
        f"TTFT {_num(d_off.get('median_ttft_ms'))} ms |"
    )
    d_on = m["decode_with_reasoning"]
    a(
        f"| Decode (reasoning on) | {_num(d_on.get('median_decode_tokens_per_second'))} tok/s | "
        f"`{d_on['reasoning_effort']}`, ~{d_on.get('median_reasoning_tokens')} reasoning tokens |"
    )
    if m["vision"].get("supported"):
        v = m["vision"]
        a(
            f"| Vision prefill | {_num(v.get('prompt_per_second'))} tok/s | "
            f"1024x768 image → {v.get('prompt_tokens')} vision tokens |"
        )
    else:
        a("| Vision prefill | not supported | server refused image input |")
    a("")

    a("## Correctness")
    a("")
    tools = m["tools"]
    a(
        f"- **Tool calling**: {'pass' if tools.get('supported') else 'FAIL'} "
        f"(`finish_reason: {tools.get('finish_reason')}`)"
    )
    for call in tools.get("tool_calls") or []:
        a(f"  - `{call.get('name')}` called with `{call.get('arguments')}`")
    if m["vision"].get("supported"):
        a(f"- **Vision reply sample**: {m['vision']['reply_sample']}")
    a("")

    a("## Memory")
    a("")
    em = m["engine_processes"]
    if em.get("available"):
        a(
            f"- Engine process tree RSS: **{em['total_rss_gib']} GiB** "
            f"({len(em['processes'])} processes). Note that Splash memory-maps the "
            "weights, so RSS understates the real working set; the engine's own "
            "`/status` memory plan is the authoritative number."
        )
    else:
        a(f"- Engine process RSS unavailable: {em.get('reason')}")
    a(f"- Host available memory: {hw.get('available_memory_gib', 'unknown')} GiB")
    a("")

    a("## Raw samples")
    a("")
    a("```json")
    a(json.dumps(round_floats(report), indent=2))
    a("```")
    a("")
    return "\n".join(lines)


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--base-url", default=DEFAULT_BASE_URL)
    ap.add_argument("--model", default=None, help="model id (default: first from /v1/models)")
    ap.add_argument("--out", default=None, help="directory for the report files")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--prefill-tokens", type=int, default=1600)
    ap.add_argument("--quick", action="store_true", help="1 repetition, shorter decode")
    args = ap.parse_args()
    if args.quick:
        args.reps = 1
        args.max_tokens = 128
        args.prefill_tokens = 800

    base_url = args.base_url.rstrip("/")

    try:
        health = http_json(f"{base_url}/health", timeout=10)
    except (urllib.error.URLError, OSError, ValueError) as exc:
        print(f"error: no server at {base_url} ({exc})", file=sys.stderr)
        return 1
    print(f"server: {base_url} {health}")

    model = args.model
    if not model:
        try:
            model = http_json(f"{base_url}/v1/models", timeout=10)["data"][0]["id"]
        except (urllib.error.URLError, OSError, ValueError, KeyError, IndexError):
            model = "bonsai-2-27b"
    print(f"model:  {model}")

    server: dict = {"base_url": base_url, "model": model}
    try:
        status = http_json(f"{base_url}/status", timeout=10)
        server["engine"] = "Splash" if "memory_plan" in status else "unknown"
        server["max_context"] = status.get("maximum_context_tokens")
        server["kv_format"] = (status.get("identity", {}).get("kv", {}) or {}).get("format")
        server["memory_plan_max_context"] = (status.get("memory_plan") or {}).get(
            "maximum_context_tokens"
        )
        plan = status.get("memory_plan") or {}
        device = plan.get("device") or {}
        if isinstance(device.get("gpu_core_count"), int):
            server["gpu_core_count"] = device["gpu_core_count"]
        if isinstance(device.get("physical_memory_bytes"), int):
            server["memory_gib"] = round(device["physical_memory_bytes"] / 1024**3, 1)
        mem = plan.get("model", {}).get("memory") or {}
        server["upstream_model"] = plan.get("model", {}).get("model_name")
        server["native_max_context"] = plan.get("model", {}).get("maximum_context_tokens")
        target = mem.get("target_weights_bytes")
        if isinstance(target, int):
            server["target_weights_gib"] = round(target / 1024**3, 2)
        draft = mem.get("draft_weights_bytes")
        if isinstance(draft, int):
            server["draft_weights_gib"] = round(draft / 1024**3, 2)
        server["memory_pressure"] = status.get("memory_pressure")
    except (urllib.error.URLError, OSError, ValueError):
        print("note:   /status unavailable (not a Splash server?) — continuing")

    measurements: dict = {}
    steps = (
        ("prefill", lambda: bench_prefill(base_url, model, args.prefill_tokens, args.reps)),
        ("decode_no_reasoning", lambda: bench_decode(base_url, model, args.max_tokens, args.reps, "none")),
        ("decode_with_reasoning", lambda: bench_decode(base_url, model, args.max_tokens, args.reps, "low")),
        ("vision", lambda: bench_vision(base_url, model)),
        ("tools", lambda: bench_tools(base_url, model)),
        ("engine_processes", engine_memory),
    )
    for name, fn in steps:
        print(f"running: {name} ...", flush=True)
        try:
            measurements[name] = fn()
        except Exception as exc:  # a failed probe must not lose the whole report
            measurements[name] = {"error": f"{type(exc).__name__}: {exc}"}
        summary = measurements[name]
        if "error" in summary:
            print(f"  -> failed: {summary['error']}", flush=True)
        elif "median_prompt_per_second" in summary:
            print(f"  -> prefill {summary['median_prompt_per_second']} tok/s", flush=True)
        elif "median_decode_tokens_per_second" in summary:
            print(f"  -> decode {summary['median_decode_tokens_per_second']} tok/s", flush=True)
        elif "supported" in summary:
            print(f"  -> {'ok' if summary['supported'] else 'unsupported'}", flush=True)
        elif "available" in summary:
            print(f"  -> {summary.get('total_rss_gib', 'n/a')} GiB RSS", flush=True)

    report = {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "benchmark_script_version": 1,
        "repetitions": args.reps,
        "decode_max_tokens": args.max_tokens,
        "server": {k: v for k, v in server.items() if v is not None},
        "hardware": hardware_facts(),
        "measurements": measurements,
    }

    out_dir = Path(args.out) if args.out else Path.cwd()
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = report["generated_at"].replace(":", "").replace("-", "").replace("T", "-")[:15]
    json_path = out_dir / f"benchmark-{stamp}.json"
    md_path = out_dir / f"benchmark-{stamp}.md"
    json_path.write_text(json.dumps(round_floats(report), indent=2) + "\n")
    md_path.write_text(render_markdown(report))

    print("")
    print(f"wrote {json_path}")
    print(f"wrote {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())