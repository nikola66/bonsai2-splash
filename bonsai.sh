#!/bin/sh
# Bonsai 2 27B — service control (Splash server API+chat page  &  Ollama bridge).
#
#   ./bonsai.sh start     load both LaunchAgents, wait for the model (~10-20 s)
#   ./bonsai.sh stop      unload both: splash serve exits, ALL model memory freed
#   ./bonsai.sh restart   stop, then start
#   ./bonsai.sh status    agent states, health, endpoints, memory
#   ./bonsai.sh install   write the LaunchAgent plists for this checkout
#   ./bonsai.sh uninstall remove the LaunchAgent plists
#
# Requires ./install-service.sh to have run once (it creates the plists this
# script loads). Both agents are RunAtLoad + KeepAlive: they come back at the
# next login once loaded; `stop` stays stopped until you `start` again.
set -u

U=$(id -u)
AG="$HOME/Library/LaunchAgents"
SERVER=com.bonsai.server
BRIDGE=com.bonsai.ollama-bridge
DEMO_DIR="$(cd "$(dirname "$0")" && pwd)"

# Shared config: bonsai.env holds the bind address and ports so these commands
# and the launchd service always agree. Values already in the environment win.
DEMO_ENV="$DEMO_DIR/bonsai.env"
if [ -f "$DEMO_ENV" ]; then
    # Values from the environment take precedence, so only set what is unset.
    _keep_host="${BONSAI_HOST:-}"; _keep_port="${PORT:-}"; _keep_bport="${BRIDGE_PORT:-}"
    . "$DEMO_ENV"
    [ -n "$_keep_host" ] && BONSAI_HOST="$_keep_host"
    [ -n "$_keep_port" ] && PORT="$_keep_port"
    [ -n "$_keep_bport" ] && BRIDGE_PORT="$_keep_bport"
fi

# Bind address the service uses. Loopback unless overridden, so an unauthenticated
# API is never published to the network by accident. Set BONSAI_HOST (or
# bonsai.env) to a LAN/VPN address to expose it — see README, "Exposing the server".
HOST="${BONSAI_HOST:-127.0.0.1}"
PORT="${PORT:-8080}"
BRIDGE_PORT="${BRIDGE_PORT:-11434}"
# A VPN/Tailscale address is a real interface, not 0.0.0.0, so it can be bound
# directly and health-checked directly.
CHECK_HOST="$HOST"
[ "$CHECK_HOST" = "0.0.0.0" ] && CHECK_HOST="127.0.0.1"

# The engine's HTTP server process for THIS service (see start_llama_server.sh):
# `python -m server.server ... --port 8080 --host=<HOST>`. It owns the port and
# the serve-native child that holds the model weights. Matching on the host and
# port keeps us from touching an unrelated Splash instance.
# Splash omits --host from the child's cmdline when it is the default
# (127.0.0.1), so only require it when we asked for something else. Arg order
# also varies (`--port 8080 --host=IP`), so accept both.
_HOST_RE="$(printf '%s' "$HOST" | sed 's/\./\\./g')"
if [ "$HOST" = "127.0.0.1" ] || [ "$HOST" = "localhost" ]; then
    SRV_PAT="server\.server .*--port[= ]${PORT}"
else
    SRV_PAT="server\.server .*--host[= ]${_HOST_RE}.*--port[= ]${PORT}|server\.server .*--port[= ]${PORT}.*--host[= ]${_HOST_RE}"
fi

loaded() { launchctl print "gui/$U/$1" >/dev/null 2>&1; }

start_one() {
    if loaded "$1"; then echo "  $1: already loaded"
    elif [ ! -f "$AG/$1.plist" ]; then
        echo "  $1: FAILED (no $AG/$1.plist — run ./install-service.sh first)"
    elif launchctl bootstrap "gui/$U" "$AG/$1.plist" 2>/dev/null; then echo "  $1: started"
    else echo "  $1: FAILED to start (check $AG/$1.plist)"; fi
}
stop_one() {
    if ! loaded "$1"; then echo "  $1: not loaded"
    elif launchctl bootout "gui/$U/$1" 2>/dev/null; then echo "  $1: stopped"
    else echo "  $1: FAILED to stop"; fi
}
server_health() {
    [ "$(curl -s -o /dev/null -w '%{http_code}' --max-time 3 "http://$CHECK_HOST:$PORT/health" 2>/dev/null)" = 200 ]
}
interface_has_host() {
    ifconfig 2>/dev/null | grep -q "inet $HOST"
}

case "${1:-status}" in
    start)
        if ! interface_has_host; then
            if [ "$HOST" = "127.0.0.1" ]; then
                echo "NOTE: loopback not listed by ifconfig; starting anyway."
            else
                echo "WARNING: bind address $HOST not found - the service binds that"
                echo "         address and will keep retrying until it exists."
                echo "         (For Tailscale: connect it, then re-run.)"
            fi
        fi
        echo "Starting:"
        start_one "$SERVER"
        start_one "$BRIDGE"
        printf "Waiting for model load"
        i=0; while [ $i -lt 60 ] && ! server_health; do printf .; i=$((i+1)); sleep 2; done
        echo
        if server_health; then
            echo "READY  demo/API:  http://$CHECK_HOST:$PORT/          (OpenAI base: /v1, any key)"
            echo "       ollama:     http://$CHECK_HOST:$BRIDGE_PORT"
        else
            echo "NOT healthy after ~120s - see logs/bonsai-server.err.log"
            exit 1
        fi
        ;;
    stop)
        echo "Stopping (releases all model memory):"
        stop_one "$SERVER"
        stop_one "$BRIDGE"
        sleep 1
        if pgrep -f "$SRV_PAT" >/dev/null 2>&1; then
            echo "WARNING: splash serve still running:"; pgrep -fl "$SRV_PAT"
        else
            echo "splash serve not running - model memory released."
            echo "Safe to use LM Studio / Ollama / other models now."
        fi
        ;;
    restart)
        "$0" stop; sleep 2; "$0" start
        ;;
    install)
        "$DEMO_DIR/install-service.sh"
        ;;
    uninstall)
        echo "Unloading and removing LaunchAgents:"
        stop_one "$SERVER"
        stop_one "$BRIDGE"
        rm -f "$AG/$SERVER.plist" "$AG/$BRIDGE.plist"
        echo "  removed $AG/$SERVER.plist"
        echo "  removed $AG/$BRIDGE.plist"
        echo "Run './bonsai.sh start' again to reinstall."
        ;;
    status)
        for l in "$SERVER" "$BRIDGE"; do
            if loaded "$l"; then
                st=$(launchctl print "gui/$U/$l" 2>/dev/null | awk '/^\t+state/{print $3; exit}')
                pid=$(launchctl print "gui/$U/$l" 2>/dev/null | awk '/^\t+pid/{print $3; exit}')
                echo "$l: loaded, state=$st pid=${pid:--}"
            else
                echo "$l: NOT loaded (stopped)"
            fi
        done
        if server_health; then
            echo "server health: OK (http://$CHECK_HOST:$PORT/health)"
        else
            echo "server health: not answering"
        fi
        bv=$(curl -s --max-time 2 "http://$CHECK_HOST:$BRIDGE_PORT/api/version" 2>/dev/null)
        [ -n "$bv" ] && echo "bridge: OK $bv" || echo "bridge: not answering"
        rpid=$(pgrep -f "$SRV_PAT" | head -1)
        if [ -n "$rpid" ]; then
            _pids="$rpid"
            for _c in $(pgrep -P "$rpid" 2>/dev/null); do _pids="$_pids,$_c"; done
            ps -o rss= -p "$_pids" | awk '{s+=$1} END {printf "splash RSS: %.1f GB (weights are file-backed; see /status for the memory plan)\n", s/1048576}'
        else
            echo "splash serve: not running (0 GB - memory free)"
        fi
        if [ "$HOST" = "127.0.0.1" ]; then
            echo "bind: loopback only (set BONSAI_HOST to expose on a LAN/VPN address)"
        elif interface_has_host; then
            echo "bind: $HOST present"
        else
            echo "bind: $HOST MISSING (service cannot bind)"
        fi
        ;;
    *)
        echo "usage: $0 start|stop|restart|status|install|uninstall"; exit 2
        ;;
esac