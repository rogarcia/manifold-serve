#!/usr/bin/env bash
# One measurement cell, from no processes to DONE. Every cell gets a new router and new engines, so
# the prefix caches and the router's index start empty.
# Usage: RUN=name cell.sh NAME FLEET_MODE SEED
#   FLEET_MODE: reconcile (floor 0) | static-2 | static-1 (autoscaler with floor=max=N, warm, trace
#     window 105 to 645) | reclaim (floor 2, warm, the reclaim trace, slot 1 reclaimed at t=60).
#   GATE: on (default) | off (no caps, no hold, no replay, as in M1).
#   TIME_SCALE (1): passed to loadgen.py. EVENTS: "T what args;..." with T in trace seconds, what
#     is kill_autoscaler or reclaim SLOT GRACE. HOLD_INTERACTIVE_S (2 x S), HOLD_BATCH_S (900),
#     SPEC_OVERRIDE (JSON of Spec fields), RECLAIM_GRACE_S (30), LOADGEN_WORKERS (8).
# Output: $M2/results/RUN/NAME/, with DONE last. A cell with DONE is skipped. A partial cell is moved
# aside and run again from the start. Needs $M2/probe.env.
set -euo pipefail
[ $# -eq 3 ] || { sed -n '2,12p' "$0" >&2; exit 2; }
NAME=$1 FLEET_MODE=$2 SEED=$3
source "$(dirname "$0")/common.sh"
RUN=${RUN:?set RUN}
GATE=${GATE:-on} TIME_SCALE=${TIME_SCALE:-1} EVENTS=${EVENTS:-}
[ -f "$M2/probe.env" ] || die "no $M2/probe.env: run probe.sh first"
source "$M2/probe.env"

TRACE=main WINDOW_ARG=()
case "$FLEET_MODE" in
  reconcile) FLOOR=0; SLOTS=2; DURATION_S=1080 ;;
  static-2)  FLOOR=2; SLOTS=2; DURATION_S=540; WINDOW_ARG=(--window static) ;;
  static-1)  FLOOR=1; SLOTS=1; DURATION_S=540; WINDOW_ARG=(--window static) ;;
  reclaim)   FLOOR=2; SLOTS=2; DURATION_S=330; TRACE=reclaim
             EVENTS=${EVENTS:-"60 reclaim 1 ${RECLAIM_GRACE_S:-30}"} ;;
  *) die "unknown fleet mode: $FLEET_MODE" ;;
esac

S=$S_FIRST
HOLD_INTERACTIVE_S=${HOLD_INTERACTIVE_S:-$(awk -v s="$S" 'BEGIN { print 2 * s }')}
HOLD_BATCH_S=${HOLD_BATCH_S:-900}
case "$GATE" in
  on)  GATE_ENV=(MANIFOLD_CAP_INTERACTIVE=96 MANIFOLD_CAP_BATCH=64
                 MANIFOLD_HOLD_INTERACTIVE_S="$HOLD_INTERACTIVE_S"
                 MANIFOLD_HOLD_BATCH_S="$HOLD_BATCH_S"
                 MANIFOLD_MAX_ATTEMPTS=3 MANIFOLD_DOWN_S=5) ;;
  off) GATE_ENV=() ;;
  *) die "GATE must be on or off" ;;
esac

if [ "$SIM" = 1 ]; then
  TOK=bytes; BLOCK=$SIM_BLOCK_TOKENS; INDEX_BLOCKS=$SIM_KV_BLOCKS
else
  TOK=$TOKENIZER_PATH; BLOCK=${BLOCK_TOKENS:?no BLOCK_TOKENS: bringup-$MODEL_TAG.env missing from $M1}
  INDEX_BLOCKS=$(( ${KV_TOKENS:?no KV_TOKENS in bringup-$MODEL_TAG.env} / BLOCK ))
fi

CELL=$M2/results/$RUN/$NAME
if [ -f "$CELL/DONE" ]; then log "skip (DONE): $CELL"; exit 0; fi
if [ -d "$CELL" ]; then mv "$CELL" "$CELL.partial-$(date -u +%Y%m%dT%H%M%SZ)"; fi
mkdir -p "$CELL"
[ -f "$DATA/sharegpt.json" ] || die "missing $DATA/sharegpt.json"
log "cell $CELL: $FLEET_MODE, gate $GATE, seed $SEED, S ${S}s, pad ${START_PAD_S}s"

cleanup() {
  local j
  for j in $(jobs -p); do kill "$j" 2>/dev/null || true; done
  stop_all
}
trap cleanup EXIT
stop_all
fresh_run_dir
ulimit -n 8192 2>/dev/null || true
require_free_port "$ROUTER_PORT"
BACKENDS=""
for (( i = 0; i < SLOTS; i++ )); do
  require_free_port "${SLOT_PORTS[$i]}"
  BACKENDS+="${BACKENDS:+,}$(slot_url "$i")"
done

SPEC_JSON=$(python "$M2_DIR/derive_spec.py" --s "$S" --floor "$FLOOR" --max-replicas "$SLOTS" \
  --override "${SPEC_OVERRIDE:-"{}"}")
echo "$SPEC_JSON" > "$CELL/spec.json"

# A warm cell starts its engines without the pad. Later starts, the reclaimed slot's included, pay it.
if [ "$FLOOR" -gt 0 ]; then echo 0 > "$PAD_FILE"; else echo "$START_PAD_S" > "$PAD_FILE"; fi

env ${GATE_ENV[@]+"${GATE_ENV[@]}"} \
  MANIFOLD_BACKENDS="$BACKENDS" MANIFOLD_POLICY=prefix_aware MANIFOLD_HOST=127.0.0.1 \
  MANIFOLD_PORT="$ROUTER_PORT" MANIFOLD_TOKENIZER="$TOK" MANIFOLD_INITIAL_STATE=off \
  MANIFOLD_CHAT_TEMPLATE_KWARGS="$CHAT_TEMPLATE_KWARGS" MANIFOLD_BLOCK_TOKENS="$BLOCK" \
  MANIFOLD_INDEX_BLOCKS="$INDEX_BLOCKS" MANIFOLD_DECISION_LOG="$CELL/decisions.jsonl" \
  python -m manifold.router > "$CELL/router.log" 2>&1 &
echo $! > "$M2_RUN_DIR/router.pid"
wait_http "$ROUTER/healthz" 180 "" "$CELL/router.log"

# Restarts the autoscaler like a supervisor would, so a kill -9 tests re-adoption.
keep_autoscaler() {
  while [ ! -f "$M2_RUN_DIR/stop-autoscaler" ]; do
    MANIFOLD_ROUTER_URL=$ROUTER MANIFOLD_SPEC=$SPEC_JSON MANIFOLD_FLEET=process \
      MANIFOLD_ENGINE_CMD="$M2_DIR/engine.sh {slot}" MANIFOLD_RUN_DIR=$M2_RUN_DIR \
      MANIFOLD_TICK_LOG=$CELL/ticks.jsonl python -m manifold.autoscaler >> "$CELL/autoscaler.log" 2>&1 &
    echo $! > "$M2_RUN_DIR/autoscaler.pid"
    wait $! || true
    sleep 1
  done
}
keep_autoscaler &

SAMPLE_TARGETS=()
for (( i = 0; i < SLOTS; i++ )); do SAMPLE_TARGETS+=("$i=$(slot_url "$i")"); done
python "$BENCH_DIR/sampler.py" "$CELL/samples.tsv" "${SAMPLE_TARGETS[@]}" &
echo $! > "$M2_RUN_DIR/sampler.pid"

if [ "$FLOOR" -gt 0 ]; then
  T0=$SECONDS
  until (( $( { curl -sf --max-time 2 "$ROUTER/stats" || true; } | grep -o '"state":"ready"' | wc -l) >= FLOOR )); do
    (( SECONDS - T0 < 600 )) || die "$FLOOR slots not READY after 600 s"
    sleep 0.5
  done
  log "warm: $FLOOR slots READY in $(( SECONDS - T0 )) s"
  echo "$START_PAD_S" > "$PAD_FILE"
fi

run_event() {
  local when=$1 what=$2 args=${3:-}
  sleep "$(awk -v t="$when" -v k="$TIME_SCALE" 'BEGIN { print t / k }')"
  printf '{"t": %s, "event": "%s", "args": "%s"}\n' "$(now)" "$what" "$args" >> "$CELL/events.jsonl"
  case "$what" in
    kill_autoscaler) kill -9 "$(cat "$M2_RUN_DIR/autoscaler.pid")" ;;
    reclaim) read -ra argv <<< "$args"; "$M2_DIR/reclaim.sh" "${argv[@]}" ;;
    *) log "unknown event: $what" ;;
  esac
}
if [ -n "$EVENTS" ]; then
  IFS=';' read -ra EVENT_LIST <<< "$EVENTS"
  for e in "${EVENT_LIST[@]}"; do
    read -r when what args <<< "$e"
    run_event "$when" "$what" "$args" >> "$CELL/events.log" 2>&1 &
  done
fi

LIMIT_S=$(awk -v d="$DURATION_S" -v k="$TIME_SCALE" -v s="$S" 'BEGIN { printf "%d\n", (d + 300) / k + 2 * s + 30 }')
rc=0
with_timeout "$LIMIT_S" python "$M2_DIR/loadgen.py" --data "$DATA/sharegpt.json" --url "$ROUTER" \
  --model "$MODEL" --out "$CELL/requests.jsonl" --trace "$TRACE" ${WINDOW_ARG[@]+"${WINDOW_ARG[@]}"} \
  --seed "$SEED" --time-scale "$TIME_SCALE" --workers "${LOADGEN_WORKERS:-8}" \
  --timeout-interactive "$(awk -v h="$HOLD_INTERACTIVE_S" 'BEGIN { print h + 300 }')" \
  --timeout-batch "$(awk -v h="$HOLD_BATCH_S" 'BEGIN { print h + 600 }')" \
  > "$CELL/loadgen.log" 2>&1 || rc=$?
# Its workers are forks with the same command line, and a timeout leaves them running.
if (( rc == 124 )); then pkill -f "$M2_DIR/loadgen.py" || true; fi

curl -sf --max-time 10 "$ROUTER/metrics" > "$CELL/router_metrics_after.txt" || true
curl -sf --max-time 10 "$ROUTER/stats" > "$CELL/router_stats_after.json" || true
trap - EXIT
cleanup
cp "$M2_RUN_DIR"/slot-*.log "$CELL/" 2>/dev/null || true
(( rc == 0 )) || { tail -20 "$CELL/loadgen.log" >&2; die "loadgen exited $rc (124 = timeout)"; }

export NAME FLEET_MODE SEED GATE S S_RESTART START_PAD_S TIME_SCALE TRACE FLOOR SLOTS CELL MODEL \
  MODEL_REVISION HOLD_INTERACTIVE_S HOLD_BATCH_S
GIT_SHA=$(cat "$REPO/GIT_SHA" 2>/dev/null || git -C "$REPO" describe --always --dirty 2>/dev/null || echo unknown)
export GIT_SHA
python - <<'EOF'
import json, os
keys = ("NAME FLEET_MODE SEED GATE S S_RESTART START_PAD_S TIME_SCALE TRACE FLOOR SLOTS MODEL "
        "MODEL_REVISION GIT_SHA SIM MODEL_TAG HOLD_INTERACTIVE_S HOLD_BATCH_S MAX_NUM_SEQS "
        "MAX_MODEL_LEN MAX_NUM_BATCHED_TOKENS GPU_MEMORY_UTILIZATION").split()
with open(os.path.join(os.environ["CELL"], "env.json"), "w") as f:
    json.dump({k: os.environ.get(k) for k in keys}, f, indent=2)
EOF
if command -v nvidia-smi >/dev/null; then nvidia-smi -L > "$CELL/gpus.txt" || true; fi
touch "$CELL/DONE"
log "done: $CELL"
