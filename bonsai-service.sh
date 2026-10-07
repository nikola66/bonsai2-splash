#!/bin/sh
# Launchd entry point for the Bonsai 2 27B OpenAI-compatible server (Splash engine).
#
# Started at login by ~/Library/LaunchAgents/com.bonsai.server.plist, which is
# installed by ./install-service.sh. Run this directly for a one-off foreground
# start; the interactive path is ./scripts/start_llama_server.sh.

# Resolve our own directory so the service works from any clone location.
DEMO_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DEMO_DIR" || exit 1

export BONSAI_FAMILY=bonsai2
export BONSAI_MODEL=27B

# Shared config with bonsai.sh (git-ignored bonsai.env, generated from
# bonsai.env.example). Environment values already set win over the file.
if [ -z "${BONSAI_HOST:-}" ] && [ -f "$DEMO_DIR/bonsai.env" ]; then
    . "$DEMO_DIR/bonsai.env"
fi

# Bind address. Defaults to loopback: this server has no authentication, so the
# network it binds to is the access control. Set BONSAI_HOST (in bonsai.env) to
# a LAN/VPN address to expose it, e.g. BONSAI_HOST="$(tailscale ip -4)".
export BONSAI_HOST="${BONSAI_HOST:-127.0.0.1}"

# Context window: the web UI writes a chosen value to .bonsai-ctx (through the
# bridge's POST /bonsai/context) and restarts the server to apply it. An
# explicit BONSAI_CTX env still wins; otherwise default to 65536.
if [ -z "${BONSAI_CTX:-}" ]; then
    BONSAI_CTX=65536
    if [ -f "$DEMO_DIR/.bonsai-ctx" ] && grep -Eq '^[0-9]+$' "$DEMO_DIR/.bonsai-ctx"; then
        _FILE_CTX="$(cat "$DEMO_DIR/.bonsai-ctx")"
        if [ "$_FILE_CTX" -ge 8192 ] && [ "$_FILE_CTX" -le 262144 ]; then
            BONSAI_CTX="$_FILE_CTX"
        fi
    fi
fi
export BONSAI_CTX

# Model: Splash serves prism-ml/Ternary-Bonsai-2-27B-gguf:PQ2_0 with its paired
# DFlash2 speculative draft (always on) and the vision projector, loaded from
# the Hugging Face cache. Override the id with BONSAI_SPLASH_MODEL.
# --alias: short model id in API dropdowns (/v1/models, chat page, bridge
# /api/tags) instead of the full repo id. The bridge still accepts any name
# (substring-matched) as a request model field. start_llama_server.sh
# translates --alias to Splash's --served-model-name + --announce-served-name.
exec ./scripts/start_llama_server.sh --alias bonsai-2-27b "$@"