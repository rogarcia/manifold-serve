# shellcheck shell=bash disable=SC2034  # the scripts that source this file use these variables
# Sourced by every bench/m2 script. Reuses the M1 environment, so MODEL_TAG, SIM, VENV and the ports
# mean the same thing here.
M2_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$M2_DIR/../m1/env.sh"
source "$BENCH_DIR/lib.sh"
export M2_DIR M2=${M2:-$WORKSPACE/m2}
# On local disk, not the Volume: a pid file must not outlive the container that wrote it.
export M2_RUN_DIR=${M2_RUN_DIR:-${TMPDIR:-/tmp}/m2-run}
export PAD_FILE=$M2_RUN_DIR/pad_s
SLOT_PORTS=("$PORT_A" "$PORT_B")
ROUTER=http://127.0.0.1:$ROUTER_PORT

slot_url() { echo "http://127.0.0.1:${SLOT_PORTS[$1]}"; }
now() { python -c 'import time; print(f"{time.time():.3f}")'; }
pad_for() { awk -v a="$1" -v b="$2" 'BEGIN { p = a - b; printf "%.1f\n", (p > 0 ? p : 0) }'; }

# Leaves the record ProcessFleet keeps: its own session, so the pid is the process group.
spawn_slot() {
  python - "$M2_DIR/engine.sh" "$1" "$M2_RUN_DIR/slot-$1.log" > "$M2_RUN_DIR/slot-$1.pid" <<'EOF'
import subprocess, sys
cmd, slot, logf = sys.argv[1:]
with open(logf, "ab") as log:
    print(subprocess.Popen([cmd, slot], stdin=subprocess.DEVNULL, stdout=log,
                           stderr=subprocess.STDOUT, start_new_session=True).pid)
EOF
}

group_alive() { kill -0 -- "-$1" 2>/dev/null; }

stop_slot() {
  local f=$M2_RUN_DIR/slot-$1.pid pgid t0=$SECONDS
  [ -f "$f" ] || return 0
  pgid=$(cat "$f")
  kill -TERM -- "-$pgid" 2>/dev/null || true
  while group_alive "$pgid" && (( SECONDS - t0 < 30 )); do sleep 0.2; done
  kill -KILL -- "-$pgid" 2>/dev/null || true
  rm -f "$f" "$M2_RUN_DIR/slot-$1.stop"
}

stop_named() {
  local f=$M2_RUN_DIR/$1.pid pid t0=$SECONDS
  [ -f "$f" ] || return 0
  pid=$(cat "$f")
  kill -TERM "$pid" 2>/dev/null || true
  while kill -0 "$pid" 2>/dev/null && (( SECONDS - t0 < 10 )); do sleep 0.2; done
  kill -KILL "$pid" 2>/dev/null || true
  rm -f "$f"
}

# Idempotent, in the order that stops anything from being restarted or left without a router.
stop_all() {
  mkdir -p "$M2_RUN_DIR"
  touch "$M2_RUN_DIR/stop-autoscaler"
  stop_named autoscaler
  stop_named sampler
  stop_slot 0
  stop_slot 1
  stop_named router
  "$BENCH_DIR/stop.sh" || log "stop.sh: an engine or GPU was not released"
}

fresh_run_dir() {
  rm -rf "${M2_RUN_DIR:?}"
  mkdir -p "$M2_RUN_DIR"
}
