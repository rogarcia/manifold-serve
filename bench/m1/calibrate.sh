#!/usr/bin/env bash
# Calibration: least_load on ShareGPT with vLLM's own KV pool, one cell per client count, then
# print p90 TTFT and waiting growth per level and write the knee and matrix loads to knee.env.
# Usage: calibrate.sh [RUN]     RUN defaults to calib; results in $M1/results/RUN/<tag>/nocap/
#   CALIB_CLIENTS (default "2 4 8 16 32 64", divisors of ACTIVE=64 so every level has exactly 64
#   live conversations), CALIB_REQ_PER_CLIENT (40), CALIB_MIN_NREQ (200),
#   CALIB_SEED (0). KNEE=N overrides the knee the rule picks. MATRIX_ACTIVE=N writes it into
#   knee.env; env.sh then uses it for ACTIVE (the matrix's live conversations) unless ACTIVE is set.
#   KNEE_FILE writes the result elsewhere, so a control model does not replace the primary's knee.env.
set -euo pipefail
RUN=${1:-calib}
source "$(dirname "$0")/env.sh"
source "$BENCH_DIR/lib.sh"

if ! { [ -f "$M1/current.env" ] && grep -qx "CUR_MODEL_TAG=$MODEL_TAG" "$M1/current.env" \
       && grep -qx "CUR_PRESSURE=nocap" "$M1/current.env"; }; then
  "$BENCH_DIR/serve.sh" "$MODEL_TAG" nocap
fi

failed=0
for c in ${CALIB_CLIENTS:-2 4 8 16 32 64}; do
  n=$(( c * ${CALIB_REQ_PER_CLIENT:-40} )); (( n >= ${CALIB_MIN_NREQ:-200} )) || n=${CALIB_MIN_NREQ:-200}
  LOAD_LABEL=calib "$BENCH_DIR/cell.sh" "$RUN" sharegpt least_load "$c" "${CALIB_SEED:-0}" "$n" \
    || { log "calibration cell c$c failed (expected above the knee if requests timed out)"; failed=$(( failed + 1 )); }
done

uv run "$BENCH_DIR/summarize.py" --calibrate "$RESULTS/$RUN/$MODEL_TAG/nocap" \
  --warmup-s "$WARMUP_S" --slo-ms "$SLO_MS" --duration-s "$DURATION_S" \
  --write-knee "${KNEE_FILE:-$M1/knee.env}" ${KNEE:+--knee "$KNEE"} ${MATRIX_ACTIVE:+--active "$MATRIX_ACTIVE"}
log "$failed calibration cells failed. Check the table, then: uv run $BENCH_DIR/size_kv.py $RESULTS/$RUN/$MODEL_TAG/nocap"
