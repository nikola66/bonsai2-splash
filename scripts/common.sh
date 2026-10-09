#!/bin/sh
# Shared helpers for Bonsai demo scripts.
# Source this file: . "$(dirname "$0")/common.sh"
#
# The demo runs on the Splash engine (https://github.com/incoai/splash), which
# serves the Bonsai 2 27B GGUF from Hugging Face with its paired DFlash2 draft.

# ── Model selection ──
# Set BONSAI_MODEL to choose size and BONSAI_FAMILY to choose family, as in
# the older demo. Splash serves Bonsai 2 27B only; any other concrete pair is
# rejected by assert_valid_model with a clear message.
BONSAI_MODEL="${BONSAI_MODEL:-27B}"
BONSAI_FAMILY="${BONSAI_FAMILY:-bonsai2}"
BONSAI_DISPLAY="Bonsai-2-${BONSAI_MODEL}"

# The model Splash serves: an upstream Hugging Face id, OWNER/REPO[:VARIANT].
# Splash downloads it (with its draft and vision projector) into its own cache
# on first run; local GGUF files are not loadable. Override with BONSAI_SPLASH_MODEL.
BONSAI_SPLASH_MODEL="${BONSAI_SPLASH_MODEL:-prism-ml/Ternary-Bonsai-2-27B-gguf:PQ2_0}"

# Validate BONSAI_MODEL + BONSAI_FAMILY — call at the top of every run/server script.
assert_valid_model() {
    case "$BONSAI_MODEL" in
        27B|all) ;;
        *)
            err "Unknown BONSAI_MODEL='${BONSAI_MODEL}'. The Splash engine serves the 27B size."
            echo "  Example: export BONSAI_MODEL=27B"
            exit 1 ;;
    esac
    case "$BONSAI_FAMILY" in
        bonsai2|all) ;;
        *)
            err "Unknown BONSAI_FAMILY='${BONSAI_FAMILY}'. The Splash engine serves the bonsai2 family (Bonsai 2 27B)."
            echo "  Example: export BONSAI_FAMILY=bonsai2"
            echo "  (The older ternary / 1-bit families need the retired llama.cpp fork.)"
            exit 1 ;;
    esac
    if [ "$BONSAI_FAMILY" != "bonsai2" ] || [ "$BONSAI_MODEL" != "27B" ]; then
        err "The Splash engine serves Bonsai 2 27B (BONSAI_FAMILY=bonsai2 BONSAI_MODEL=27B)."
        echo "  Got: BONSAI_FAMILY=${BONSAI_FAMILY} BONSAI_MODEL=${BONSAI_MODEL}"
        echo "  'all' is only meaningful for downloads; pick a concrete supported pair."
        exit 1
    fi
}

# ── Colors ──
if [ -t 1 ]; then
    _CLR_GREEN="\033[32m"
    _CLR_YELLOW="\033[33m"
    _CLR_RED="\033[31m"
    _CLR_CYAN="\033[36m"
    _CLR_RESET="\033[0m"
else
    _CLR_GREEN="" _CLR_YELLOW="" _CLR_RED="" _CLR_CYAN="" _CLR_RESET=""
fi

info()  { printf "${_CLR_GREEN}[OK]${_CLR_RESET}   %s\n" "$*"; }
warn()  { printf "${_CLR_YELLOW}[WARN]${_CLR_RESET} %s\n" "$*"; }
err()   { printf "${_CLR_RED}[ERR]${_CLR_RESET}  %s\n" "$*" >&2; }
step()  { printf "${_CLR_CYAN}==>    %s${_CLR_RESET}\n" "$*"; }

# ── Locate the Splash binary ──
# launchd's PATH is minimal (/usr/bin:/bin:/usr/sbin:/sbin), so a Homebrew
# install in /opt/homebrew/bin is not guaranteed to be found via command -v.
splash_bin() {
    _sb="$(command -v splash 2>/dev/null || true)"
    if [ -z "$_sb" ]; then
        for _c in /opt/homebrew/bin/splash /usr/local/bin/splash "$HOME/.local/bin/splash"; do
            [ -x "$_c" ] && _sb="$_c" && break
        done
    fi
    echo "$_sb"
}

# ── Default context: a predictable RAM-tiered cap (same tiers as the old demo).
# Splash sizes context automatically up to the model's native 262144-token
# window when memory allows; we still pass an explicit --max-context so memory
# use stays predictable on this machine. Override with BONSAI_CTX=N
# (up to 262144). BONSAI_CTX=0 or unset both mean "auto" and resolve to the
# tiered default below; to force full training context pass e.g. BONSAI_CTX=262144.
#
# The 24 GB+ tier is 65536, not 32768: agent harnesses (Hermes) refuse a model
# under a 64K context, and coding agents want the headroom. 24 GB pays ~1 GB
# more KV for it (int8), well within the machine's plan.
bonsai_ctx_default() {
    # Treat 0 the same as unset ("auto"): never emit -c 0 downstream.
    if [ -n "${BONSAI_CTX:-}" ] && [ "$BONSAI_CTX" != "0" ]; then
        echo "$BONSAI_CTX"
        return
    fi
    if [ "$(uname -s)" = "Darwin" ]; then
        _mem_gb=$(( $(sysctl -n hw.memsize) / 1073741824 ))
    else
        _mem_kb=$(awk '/MemTotal/ {print $2}' /proc/meminfo 2>/dev/null)
        _mem_gb=$(( ${_mem_kb:-0} / 1048576 ))
    fi
    if [ "$_mem_gb" -le 11 ] 2>/dev/null; then
        echo 8192
    elif [ "$_mem_gb" -le 23 ] 2>/dev/null; then
        echo 16384
    elif [ "$_mem_gb" -le 71 ] 2>/dev/null; then
        echo 65536
    else
        echo 131072
    fi
}
CTX_SIZE_DEFAULT=$(bonsai_ctx_default)

# ── Image token cap for the vision model, in Splash's --max-image-pixels.
# 1 vision token is 32x32 px = 1024 px, so tokens x 1024 = pixels. The old
# demo capped images at 1024 tokens on consumer hardware (fast, fine detail in
# big images lost); Splash's own default (4194304 px = 4096 tokens) is the
# uncapped equivalent. Override with BONSAI_IMAGE_MAX_TOKENS (0 = uncapped).
bonsai_image_max_pixels() {
    if [ -n "${BONSAI_IMAGE_MAX_TOKENS:-}" ]; then
        [ "$BONSAI_IMAGE_MAX_TOKENS" = "0" ] || echo $(( BONSAI_IMAGE_MAX_TOKENS * 1024 ))
    else
        echo 1048576  # 1024 vision tokens, the old consumer-hardware default
    fi
}

# ── Resolve DEMO_DIR (parent of scripts/) ──
resolve_demo_dir() {
    _script_dir="$(cd "$(dirname "$0")" && pwd)"
    echo "$(cd "$_script_dir/.." && pwd)"
}
