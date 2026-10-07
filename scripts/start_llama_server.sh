#!/bin/sh
# Start an OpenAI-compatible chat server with the Bonsai model (Splash engine).
# Usage: ./scripts/start_llama_server.sh
# Then open http://localhost:8080 in your browser.
#
# Same entry point and environment variables as the old llama.cpp launcher;
# llama.cpp-only flags are translated to their Splash equivalents or ignored
# with a note, and any Splash-native `splash serve` flag passes straight
# through (e.g. --max-memory 20G, --kv-format bf16, --language-only).
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
. "$SCRIPT_DIR/common.sh"
assert_valid_model
cd "$(resolve_demo_dir)"

# Shared config (git-ignored bonsai.env, generated from bonsai.env.example).
# Sourced here too so a plain `./scripts/start_llama_server.sh` honours it;
# values already in the environment take precedence.
DEMO_DIR="$(resolve_demo_dir)"
if [ -f "$DEMO_DIR/bonsai.env" ]; then
    _keep_host="${BONSAI_HOST:-}"; _keep_port="${PORT:-}"
    . "$DEMO_DIR/bonsai.env"
    [ -n "$_keep_host" ] && BONSAI_HOST="$_keep_host"
    [ -n "$_keep_port" ] && PORT="$_keep_port"
fi

# Bind to loopback by default; override with BONSAI_HOST=0.0.0.0 for LAN/remote.
# There is no authentication, so the bind address is the access control.
HOST="${BONSAI_HOST:-127.0.0.1}"
PORT="${PORT:-8080}"

# ── Find the Splash binary (launchd's PATH may miss Homebrew) ──
SPLASH="$(splash_bin)"
if [ -z "$SPLASH" ]; then
    err "splash not found."
    echo "  Install it with:  brew install incoai/tap/splash"
    exit 1
fi

# ── Check the port is free ──
_CHECK_HOST="$HOST"
[ "$_CHECK_HOST" = "0.0.0.0" ] && _CHECK_HOST="127.0.0.1"
if curl -s --max-time 2 "http://$_CHECK_HOST:$PORT/health" >/dev/null 2>&1; then
    warn "A server is already running on port $PORT."
    echo "  Stop it first with:  kill \$(lsof -ti TCP:$PORT)"
    exit 1
fi

# ── Base configuration (env-driven, same knobs as before) ──
MODEL="$BONSAI_SPLASH_MODEL"
CTX="$CTX_SIZE_DEFAULT"
_IMG="$(bonsai_image_max_pixels)"          # BONSAI_IMAGE_MAX_TOKENS -> pixels
_IDLE="${BONSAI_IDLE_RELEASE:-off}"        # keep the model loaded (old behavior)
_EFFORT="low"   # default thinking level when no reasoning flag is supplied
SPLASH_ARGS=""
_ANNOUNCE=0
_DROPPED=""
_NOTE=""

_note_dropped() { _DROPPED="${_DROPPED:+$_DROPPED }$1"; }

# New Splash-specific knobs (optional).
[ -n "${BONSAI_KV_FORMAT:-}" ]    && SPLASH_ARGS="$SPLASH_ARGS --kv-format $BONSAI_KV_FORMAT"
[ -n "${BONSAI_MAX_MEMORY:-}" ]   && SPLASH_ARGS="$SPLASH_ARGS --max-memory $BONSAI_MAX_MEMORY"
[ -n "${BONSAI_MAX_CACHE_DISK:-}" ] && SPLASH_ARGS="$SPLASH_ARGS --max-cache-disk $BONSAI_MAX_CACHE_DISK"
case "${BONSAI_LANGUAGE_ONLY:-0}" in 1|true|yes|on) SPLASH_ARGS="$SPLASH_ARGS --language-only" ;; esac
case "${BONSAI_OFFLINE:-0}" in 1|true|yes|on) SPLASH_ARGS="$SPLASH_ARGS --offline" ;; esac

# Legacy llama.cpp-only env vars: accepted so old configs don't break, noted once.
for _v in BONSAI_GGUF BONSAI_MMPROJ BONSAI_NGL BONSAI_SPECULATIVE BONSAI_MMPROJ_CPU BONSAI_LLAMA_VERBOSE; do
    eval "_val=\${$_v:-}"
    [ -n "$_val" ] && [ "$_val" != "0" ] && _note_dropped "$_v"
done
if [ -n "${BONSAI_KV4:-}" ] && [ "${BONSAI_KV4}" != "0" ]; then
    _NOTE="KV cache is int8 in Splash by default (~half the old FP16 memory); pass --kv-format bf16 for full precision."
fi

# ── Translate legacy llama.cpp flags to Splash flags ──
# Rotate idiom: each original argument is shifted once; Splash-native flags are
# moved to the back of "$@" (kept in order), translated ones build SPLASH_ARGS,
# llama.cpp-only ones are consumed (flag + value) and reported once.
_PRE=""   # non-empty = the NEXT argument is the value of the flag just seen
_N=$#
_I=0
while [ "$_I" -lt "$_N" ]; do
    _A="$1"
    shift
    _I=$((_I + 1))

    # Value of the previous flag, whatever it looks like.
    if [ -n "$_PRE" ]; then
        case "$_PRE" in
            ctx)    CTX="$_A" ;;
            model)  MODEL="$_A" ;;
            alias)  SPLASH_ARGS="$SPLASH_ARGS --served-model-name $_A"; _ANNOUNCE=1 ;;
            img)    if [ "$_A" = "0" ]; then _IMG=""; else _IMG=$((_A * 1024)); fi ;;
            effort)
                case "$_A" in
                    0) _EFFORT="none" ;;
                    -1) _EFFORT="" ;;  # unlimited = the engine's model default (no flag)
                    *)
                        if [ "$_A" -le 512 ] 2>/dev/null; then _EFFORT="low"
                        elif [ "$_A" -le 2048 ] 2>/dev/null; then _EFFORT="medium"
                        elif [ "$_A" -le 8192 ] 2>/dev/null; then _EFFORT="high"
                        else _EFFORT="max"; fi
                        warn "reasoning budget $_A mapped to reasoning effort '$_EFFORT' (Splash budgets are per-request reasoning_effort)." ;;
                esac ;;
            tpl)
                case "$_A" in *'"enable_thinking": false'*|*"enable_thinking\": false"*) _EFFORT="none" ;; esac ;;
            drop)   : ;;
        esac
        _PRE=""
        continue
    fi

    case "$_A" in
        # ── translated ──
        -c|--ctx-size)     _PRE="ctx" ;;
        -m|--model)        _PRE="model" ;;
        --alias)           _PRE="alias" ;;
        --image-max-tokens) _PRE="img" ;;
        --reasoning-budget) _PRE="effort" ;;
        --chat-template-kwargs) _PRE="tpl"; _note_dropped "--chat-template-kwargs" ;;
        # ── llama.cpp-only: flag + value, ignored ──
        --reasoning-format|--webui-config-file|--mmproj|--spec-type|\
        --spec-draft-n-max|-md|-ngld|-np|--parallel|--cache-type-k|\
        --cache-type-v|--kv-mean-center|--cache-reuse|-ub|-ngl|-fa|\
        --temp|--top-p|--top-k|--min-p)
            _note_dropped "$_A"; _PRE="drop" ;;
        # ── llama.cpp-only: bare flags, ignored ──
        --jinja|--no-mmproj-offload|-v|--verbose|--reasoning-budget-message)
            _note_dropped "$_A" ;;
        # ── everything else: Splash-native (or a typo Splash will name) ──
        *)  set -- "$@" "$_A" ;;
    esac
done
if [ -n "$_PRE" ]; then
    warn "flag at end of command line is missing its value; dropped."
fi
[ "$_ANNOUNCE" = "1" ] && SPLASH_ARGS="$SPLASH_ARGS --announce-served-name"
[ -n "$_DROPPED" ] && warn "ignored llama.cpp-only settings: $_DROPPED"
[ -n "$_NOTE" ] && echo "  Note: $_NOTE"

if [ ! -d "$HOME/Library/Application Support/Splash/models" ]; then
    echo "  Note: first run downloads the model + draft (~10 GB) from Hugging Face."
fi

echo ""
echo "=== Splash server (Bonsai 2 27B) ==="
echo "  Model:   $MODEL"
echo "  Engine:  $SPLASH"
[ "${BONSAI_LANGUAGE_ONLY:-0}" = "1" ] \
    || echo "  Vision:  bundled with the model repo (images accepted)"
echo "  Context: $CTX tokens (override with BONSAI_CTX, 0 = auto)"
if [ -n "$_EFFORT" ]; then
    echo "  Thinking: default reasoning effort '$_EFFORT'"
else
    echo "  Thinking: engine model default (no reasoning effort flag)"
fi
echo ""
echo "  Open http://localhost:$PORT in your browser to chat."
echo "  API:  http://localhost:$PORT/v1/chat/completions"
echo "  Press Ctrl+C to stop."
echo ""

# Run Splash as a child process — NOT exec. launchd (bootout) and the terminal
# (Ctrl+C) signal THIS script; the trap stops the whole engine tree. A plain
# SIGTERM to the `splash serve` wrapper alone would orphan the `server.server`
# HTTP process that holds the port (Splash spawns it; it spawns serve-native).
# shellcheck disable=SC2086
"$SPLASH" serve \
    --model "$MODEL" \
    --host "$HOST" \
    --port "$PORT" \
    --max-context "$CTX" \
    ${_IMG:+--max-image-pixels $_IMG} \
    ${_EFFORT:+--default-reasoning-effort $_EFFORT} \
    --idle-release "$_IDLE" \
    $SPLASH_ARGS \
    "$@" &
CLI=$!

_stop_tree() {
    trap - TERM INT
    # server.server with our host/port = this instance's HTTP server; its
    # graceful shutdown also stops the serve-native engine child (the weights).
    pkill -TERM -f "server\.server .*--host[= ]${HOST}.*--port[= ]${PORT}" 2>/dev/null || :
    pkill -TERM -f "server\.server .*--port[= ]${PORT}.*--host[= ]${HOST}" 2>/dev/null || :
    kill -TERM "$CLI" 2>/dev/null || :
    wait "$CLI" 2>/dev/null || :
    exit 0
}
trap '_stop_tree' TERM INT

_RC=0
wait "$CLI" || _RC=$?
# The wrapper exited on its own (crash or external kill): make sure no HTTP
# server for this instance outlives it holding the port.
pkill -TERM -f "server\.server .*--host[= ]${HOST}.*--port[= ]${PORT}" 2>/dev/null || :
pkill -TERM -f "server\.server .*--port[= ]${PORT}.*--host[= ]${HOST}" 2>/dev/null || :
exit "$_RC"
