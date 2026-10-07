#!/bin/sh
# Generate the launchd agents for this checkout and load them.
#
#   ./install-service.sh            install + load (starts the model)
#   ./install-service.sh --no-load  only write the plists
#
# The plists embed this checkout's absolute path and Python interpreter, so
# re-run this after moving or renaming the directory. Uninstall with
# ./bonsai.sh uninstall.
set -eu

DEMO_DIR="$(cd "$(dirname "$0")" && pwd)"
AG="$HOME/Library/LaunchAgents"
U=$(id -u)
SERVER_LABEL=com.bonsai.server
BRIDGE_LABEL=com.bonsai.ollama-bridge
LOAD=1
[ "${1:-}" = "--no-load" ] && LOAD=0

[ "$(uname -s)" = "Darwin" ] || { echo "This installer is macOS-only (launchd)."; exit 1; }

# The bridge needs httpx, so it must run with an interpreter that has it.
# Prefer the project venv; otherwise look for one that can import httpx already.
pick_python() {
    # Iterated with `read`, not `for`, so paths containing spaces survive.
    _candidates="$DEMO_DIR/.venv/bin/python
$(command -v python3 2>/dev/null || true)
$(command -v python3.12 2>/dev/null || true)
$(command -v python3.11 2>/dev/null || true)"
    printf '%s\n' "$_candidates" | while IFS= read -r _c; do
        [ -n "$_c" ] || continue
        [ -x "$_c" ] || continue
        if "$_c" -c "import httpx" >/dev/null 2>&1; then
            echo "$_c"
            break
        fi
    done
}

# Check the value, not just the exit status: `pick_python` runs in a pipeline
# subshell, so it can print nothing while still exiting 0.
PY="$(pick_python)"
if [ -z "$PY" ]; then
    echo "ERROR: no Python 3 with httpx found in $DEMO_DIR/.venv or on PATH." >&2
    echo "       Create the venv first:" >&2
    echo "         python3 -m venv .venv && .venv/bin/pip install httpx" >&2
    exit 1
fi
echo "bridge interpreter: $PY"

mkdir -p "$DEMO_DIR/logs" "$AG"

cat > "$AG/$SERVER_LABEL.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key>
  <string>$SERVER_LABEL</string>
  <key>ProgramArguments</key>
  <array>
    <string>$PY</string>
    <string>-c</string>
    <string>import os; os.execv('/bin/sh', ['/bin/sh', '$DEMO_DIR/bonsai-service.sh'])</string>
  </array>
  <key>WorkingDirectory</key>
  <string>$DEMO_DIR</string>
  <key>RunAtLoad</key>
  <true/>
  <key>KeepAlive</key>
  <true/>
  <key>ThrottleInterval</key>
  <integer>30</integer>
  <key>ProcessType</key>
  <string>Standard</string>
  <key>StandardOutPath</key>
  <string>$DEMO_DIR/logs/bonsai-server.log</string>
  <key>StandardErrorPath</key>
  <string>$DEMO_DIR/logs/bonsai-server.err.log</string>
</dict>
</plist>
PLIST

cat > "$AG/$BRIDGE_LABEL.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>$BRIDGE_LABEL</string>
    <key>ProgramArguments</key>
    <array>
        <string>$PY</string>
        <string>$DEMO_DIR/ollama_bridge.py</string>
    </array>
    <key>WorkingDirectory</key>
    <string>$DEMO_DIR</string>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
    <key>ThrottleInterval</key>
    <integer>10</integer>
    <key>ProcessType</key>
    <string>Standard</string>
    <key>StandardOutPath</key>
    <string>$DEMO_DIR/logs/ollama-bridge.log</string>
    <key>StandardErrorPath</key>
    <string>$DEMO_DIR/logs/ollama-bridge.err.log</string>
</dict>
</plist>
PLIST

echo "wrote $AG/$SERVER_LABEL.plist"
echo "wrote $AG/$BRIDGE_LABEL.plist"

if [ "$LOAD" = "1" ]; then
    # Replacing a plist while its agent is loaded does not take effect, so
    # unload first (bonsai.sh stop, which also reaps the engine tree) and wait
    # for the agents to actually leave the launchd domain before restarting.
    "$DEMO_DIR/bonsai.sh" stop >/dev/null 2>&1 || true
    for _ in 1 2 3 4 5 6 7 8 9 10; do
        still=0
        for l in "$SERVER_LABEL" "$BRIDGE_LABEL"; do
            launchctl print "gui/$U/$l" >/dev/null 2>&1 && still=1
        done
        [ "$still" = "0" ] && break
        sleep 1
    done
    echo ""
    exec "$DEMO_DIR/bonsai.sh" start
else
    echo ""
    echo "Start it with:  ./bonsai.sh start"
fi