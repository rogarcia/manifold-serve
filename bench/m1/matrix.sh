#!/usr/bin/env bash
# Main matrix: pressure outer (replicas restart once per pressure), then seed, load, and the four
# policies with their order rotated by seed. Cells with DONE are skipped, so rerunning resumes.
# Usage: matrix.sh RUN
#   PRESSURES ("fits pressure"), SEEDS ("1 2"), LOADS ("30 70 95", percent of the knee),
#   WORKLOAD (sharegpt), MAX_CONSEC_FAIL (2). MODEL_TAG picks the model (q35-9b).
#   NREQ replaces the request count from knee.env.
#   Client counts and request counts per load come from $M1/knee.env (summarize.py --calibrate).
set -euo pipefail
RUN=${1:?usage: matrix.sh RUN}
source "$(dirname "$0")/env.sh"
source "$BENCH_DIR/lib.sh"
[ -f "$M1/knee.env" ] || die "no $M1/knee.env: run calibrate.sh first"

PRESSURES=${PRESSURES:-fits pressure}
SEEDS=${SEEDS:-1 2}
LOADS=${LOADS:-30 70 95}
WORKLOAD=${WORKLOAD:-sharegpt}
POLICIES=(round_robin least_load session_hash prefix_aware)

# Lines "seed load policy" in run order for one pressure.
plan() {
  local i=0 s f k
  for s in $SEEDS; do
    for f in $LOADS; do
      for k in 0 1 2 3; do echo "$s $f ${POLICIES[$(( (k + i) % 4 ))]}"; done
    done
    i=$(( i + 1 ))
  done
}
clients_for() { local v=CLIENTS_$1; [ -n "${!v:-}" ] || die "$v missing from knee.env"; echo "${!v}"; }
nreq_for() { local v=NREQ_$1; [ -n "${!v:-}" ] || die "$v missing from knee.env"; echo "${!v}"; }
cell_dir() { echo "$RESULTS/$RUN/$MODEL_TAG/$1/$WORKLOAD/$4-c$(clients_for "$3")-s$2"; }

for f in $LOADS; do
  c=$(clients_for "$f"); r=$(nreq_for "$f"); per=$(( (ACTIVE + c / 2) / c )); (( per >= 1 )) || per=1
  log "load $f%: $c clients, $r requests, $(( per * c )) live conversations (ACTIVE $ACTIVE)"
  if (( (per * c - ACTIVE) * 100 > 15 * ACTIVE || (ACTIVE - per * c) * 100 > 15 * ACTIVE )); then
    log "WARNING: load $f% runs $(( per * c )) live conversations, more than 15% away from ACTIVE=$ACTIVE"
  fi
done

total=0; failed=0; consec=0
for p in $PRESSURES; do
  todo=0
  while read -r s f k; do [ -f "$(cell_dir "$p" "$s" "$f" "$k")/DONE" ] || todo=$(( todo + 1 )); done < <(plan)
  if (( todo == 0 )); then log "pressure $p: all cells done"; continue; fi
  if ! { [ -f "$M1/current.env" ] && grep -qx "CUR_MODEL_TAG=$MODEL_TAG" "$M1/current.env" \
         && grep -qx "CUR_PRESSURE=$p" "$M1/current.env" && curl -sf "$URL_A/health" >/dev/null \
         && curl -sf "$URL_B/health" >/dev/null; }; then
    "$BENCH_DIR/serve.sh" "$MODEL_TAG" "$p"
  fi
  n=0
  while read -r s f k; do
    n=$(( n + 1 )); total=$(( total + 1 ))
    log "[$p $n/$(plan | wc -l | tr -d ' ')] seed $s load $f% $k"
    if LOAD_LABEL=$f "$BENCH_DIR/cell.sh" "$RUN" "$WORKLOAD" "$k" "$(clients_for "$f")" "$s" "${NREQ:-$(nreq_for "$f")}" </dev/null; then
      consec=0
    else
      failed=$(( failed + 1 )); consec=$(( consec + 1 ))
      echo "$(date -u +%FT%TZ) $RUN $MODEL_TAG $p $WORKLOAD $k c$(clients_for "$f") s$s" >> "$LOGS/failed-cells.txt"
      (( consec < ${MAX_CONSEC_FAIL:-2} )) || die "$consec cells in a row failed; check the replicas (logs in $LOGS)"
    fi
  done < <(plan)
done
log "matrix $RUN: $total cells attempted, $failed failed"
(( failed == 0 ))
