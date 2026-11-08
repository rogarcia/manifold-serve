#!/usr/bin/env bash
# Measure what a start costs and check the two engine behaviours the gate depends on. Run first in
# a fresh container: S_FIRST is only the fresh-container cost there.
# Writes $M2/probe.env with S_FIRST, S_RESTART, START_PAD_S and exits non-zero on a failed check, so
# no cell runs on an engine that keeps generating after a preemption.
#   cancel: closing the client connection of a stream:false request stops the generation.
#   drain:  a stream running at SIGTERM to the engine's pid, the signal the fleet sends, still finishes.
# Usage: probe.sh     RUN is recorded as PROBE_RUN. PROBE_DRAIN_TOKENS (600).
set -euo pipefail
source "$(dirname "$0")/common.sh"
[ -x "$VENV/bin/python" ] || die "no venv at $VENV"
CANCEL_TOKENS=2000

trap stop_all EXIT
stop_all
fresh_run_dir
echo 0 > "$PAD_FILE"
for p in "${SLOT_PORTS[@]}"; do require_free_port "$p"; done

time_start() {
  local slot=$1 t0 url pgid
  t0=$(now); url=$(slot_url "$slot")
  spawn_slot "$slot"
  pgid=$(cat "$M2_RUN_DIR/slot-$slot.pid")
  until curl -sf --max-time 2 "$url/health" >/dev/null; do
    group_alive "$pgid" || { tail -40 "$M2_RUN_DIR/slot-$slot.log" >&2; die "slot $slot exited during start"; }
    (( SECONDS < 1200 )) || die "slot $slot: no /health after 20 minutes"
    sleep 0.2
  done
  awk -v a="$(now)" -v b="$t0" 'BEGIN { printf "%.1f\n", a - b }'
}

running() { { curl -sf --max-time 5 "$1/metrics" || true; } | metric_sum vllm:num_requests_running; }

chat_body() {
  python -c 'import json, sys
print(json.dumps({"model": sys.argv[1], "max_tokens": int(sys.argv[2]), "ignore_eos": True,
                  "stream": sys.argv[3] == "stream",
                  "messages": [{"role": "user", "content": "Count upward from one, forever."}]}))' \
    "$MODEL" "$1" "$2"
}

check_cancel() {
  local url=$1 body t0 pids=() p n
  body=$(chat_body "$CANCEL_TOKENS" whole)
  for _ in 1 2 3 4; do
    curl -s -o /dev/null --max-time 120 -H 'Content-Type: application/json' -d "$body" \
      "$url/v1/chat/completions" &
    pids+=($!)
  done
  t0=$SECONDS
  until (( $(running "$url") >= 4 )); do
    (( SECONDS - t0 < 30 )) || { log "cancel: 4 requests never ran on $url"; return 1; }
    sleep 0.2
  done
  kill "${pids[@]}" 2>/dev/null || true
  for p in "${pids[@]}"; do wait "$p" 2>/dev/null || true; done
  t0=$SECONDS
  until n=$(running "$url"); (( n == 0 )); do
    (( SECONDS - t0 < 8 )) || { log "cancel: running still $n, 8 s after the clients left"; return 1; }
    sleep 0.2
  done
  log "cancel ok: running fell to 0 in $(( SECONDS - t0 )) s"
}

check_drain() {
  local slot=$1 url out body t0 pgid curl_pid
  url=$(slot_url "$slot"); out=$M2_RUN_DIR/drain.sse
  pgid=$(cat "$M2_RUN_DIR/slot-$slot.pid")
  body=$(chat_body "${PROBE_DRAIN_TOKENS:-600}" stream)
  curl -sN --max-time 120 -H 'Content-Type: application/json' -d "$body" \
    "$url/v1/chat/completions" > "$out" &
  curl_pid=$!
  t0=$SECONDS
  until [ -s "$out" ]; do
    (( SECONDS - t0 < 30 )) || { log "drain: no first token"; return 1; }
    sleep 0.1
  done
  kill -TERM "$pgid"
  wait "$curl_pid" || true
  grep -q '\[DONE\]' "$out" || { log "drain: the stream was cut at SIGTERM"; return 1; }
  t0=$SECONDS
  while group_alive "$pgid"; do
    (( SECONDS - t0 < 30 )) || { log "drain: engine alive 30 s after the stream ended"; return 1; }
    sleep 0.2
  done
  log "drain ok: stream finished after SIGTERM, engine exited"
}

S_FIRST=$(time_start 0)
log "slot 0 start: ${S_FIRST}s"
S_SECOND=$(time_start 1)
log "slot 1 start: ${S_SECOND}s"
stop_slot 1
S_RESTART=$(time_start 1)
log "slot 1 restart: ${S_RESTART}s"
START_PAD_S=$(pad_for "$S_FIRST" "$S_RESTART")

check_cancel "$(slot_url 0)" || die "the engine keeps generating after a client disconnect"
check_drain 1 || die "--shutdown-timeout did not drain a running stream"

mkdir -p "$M2"
cat > "$M2/probe.env.tmp" <<EOF
PROBE_RUN=${RUN:-}
S_FIRST=$S_FIRST
S_SECOND=$S_SECOND
S_RESTART=$S_RESTART
START_PAD_S=$START_PAD_S
EOF
mv "$M2/probe.env.tmp" "$M2/probe.env"
log "probe: S_FIRST=$S_FIRST S_RESTART=$S_RESTART START_PAD_S=$START_PAD_S"
