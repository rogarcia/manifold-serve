#!/usr/bin/env bash
# Run a fixed list of sharegpt cells in priority order, and never start a cell (plus the restart
# it needs) that the running time estimates say cannot finish before the deadline. A skipped line
# does not stop the run: a later line that needs no restart may still fit. Cells with DONE are
# skipped, so rerunning resumes. Stops the replicas on exit. Replaces matrix.sh for a capped run:
# the 2026-10-24 budget ceiling is about 2 pod-hours.
# Usage: BUDGET_MIN=110 budget.sh RUN
#   BUDGET_MIN (required): minutes from now to the deadline.
#   PLAN_FILE: lines "pressure load policy seed" in run order, replacing the plan below
#     (blank lines and # comments ignored).
#   CELL_EST_S (330), RESTART_EST_S (330): starting estimates, each replaced by the longest successful cell
#     and the longest serve.sh restart seen so far.
#   MAX_CONSEC_FAIL (2). MODEL_TAG picks the model (q35-9b).
#   Client counts and request counts per load come from $M1/knee.env (summarize.py --calibrate).
set -euo pipefail
RUN=${1:?usage: BUDGET_MIN=minutes budget.sh RUN}
[ -n "${BUDGET_MIN:-}" ] || { sed -n '7,14p' "$0" >&2; exit 2; }
source "$(dirname "$0")/env.sh"
source "$BENCH_DIR/lib.sh"
[ -f "$M1/knee.env" ] || die "no $M1/knee.env: run calibrate.sh first"
[[ "$BUDGET_MIN" =~ ^[0-9]+$ ]] || die "BUDGET_MIN must be whole minutes"

deadline=$(( SECONDS + BUDGET_MIN * 60 ))
# Longest cell and restart seen so far; empty until one has run.
cell_max=""; restart_max=""

plan() {
  if [ -n "${PLAN_FILE:-}" ]; then sed 's/#.*//' "$PLAN_FILE"; else cat <<'EOF'
pressure 95 least_load    1
pressure 95 prefix_aware  1
pressure 95 session_hash  1
pressure 95 round_robin   1
pressure 95 round_robin   2
pressure 95 session_hash  2
pressure 95 prefix_aware  2
pressure 95 least_load    2
fits     95 least_load    1
fits     95 prefix_aware  1
pressure 70 least_load    1
pressure 70 prefix_aware  1
EOF
  fi | awk 'NF'
}
clients_for() { local v=CLIENTS_$1; [ -n "${!v:-}" ] || die "$v missing from knee.env"; echo "${!v}"; }
nreq_for() { local v=NREQ_$1; [ -n "${!v:-}" ] || die "$v missing from knee.env"; echo "${!v}"; }
# What serves now, checked fresh before every line so a replica that died mid-run gets restarted.
serving() {
  if [ -f "$M1/current.env" ] && grep -qx "CUR_MODEL_TAG=$MODEL_TAG" "$M1/current.env" \
     && curl -sf "$URL_A/health" >/dev/null && curl -sf "$URL_B/health" >/dev/null; then
    sed -n 's/^CUR_PRESSURE=//p' "$M1/current.env"
  fi
}

# Read the whole plan first so a bad line fails before any GPU time is spent, and so nothing in
# the loop reads the plan from stdin.
LINES=()
while read -r p f k s extra; do
  [ -z "${extra:-}" ] && [ -n "${s:-}" ] || die "plan line needs 4 fields: $p $f $k ${s:-} ${extra:-}"
  case "$p" in fits|pressure) ;; *) die "unknown pressure in plan: $p" ;; esac
  case "$k" in round_robin|least_load|session_hash|prefix_aware) ;; *) die "unknown policy in plan: $k" ;; esac
  clients_for "$f" >/dev/null; nreq_for "$f" >/dev/null
  LINES+=("$p $f $k $s")
done < <(plan)
(( ${#LINES[@]} > 0 )) || die "empty plan"
trap '"$BENCH_DIR/stop.sh"' EXIT

ran=0; done_=0; over=0; failed=0; consec=0; n=0
for line in "${LINES[@]}"; do
  read -r p f k s <<< "$line"
  n=$(( n + 1 ))
  c=$(clients_for "$f")
  if [ -f "$RESULTS/$RUN/$MODEL_TAG/$p/sharegpt/$k-c$c-s$s/DONE" ]; then
    log "[$n/${#LINES[@]}] skip (DONE): $line"; done_=$(( done_ + 1 )); continue
  fi
  cur=$(serving)
  cost=${cell_max:-${CELL_EST_S:-330}}
  [ "$cur" = "$p" ] || cost=$(( cost + ${restart_max:-${RESTART_EST_S:-330}} ))
  left=$(( deadline - SECONDS ))
  if (( cost > left )); then
    log "[$n/${#LINES[@]}] skip (budget): $line, needs ${cost}s, ${left}s left"; over=$(( over + 1 )); continue
  fi
  if [ "$cur" != "$p" ]; then
    t0=$SECONDS
    "$BENCH_DIR/serve.sh" "$MODEL_TAG" "$p" </dev/null
    d=$(( SECONDS - t0 )); if [ -z "$restart_max" ] || (( d > restart_max )); then restart_max=$d; fi
  fi
  log "[$n/${#LINES[@]}] $line ($(( deadline - SECONDS ))s left)"
  t0=$SECONDS; rc=0
  LOAD_LABEL=$f "$BENCH_DIR/cell.sh" "$RUN" sharegpt "$k" "$c" "$s" "$(nreq_for "$f")" </dev/null || rc=$?
  d=$(( SECONDS - t0 ))
  # A cell that fails early says nothing about how long a full one takes.
  if (( rc == 0 )) && { [ -z "$cell_max" ] || (( d > cell_max )); }; then cell_max=$d; fi
  if (( rc == 0 )); then
    ran=$(( ran + 1 )); consec=0
  else
    failed=$(( failed + 1 )); consec=$(( consec + 1 ))
    echo "$(date -u +%FT%TZ) $RUN $MODEL_TAG $p sharegpt $k c$c s$s" >> "$LOGS/failed-cells.txt"
    if (( consec >= ${MAX_CONSEC_FAIL:-2} )); then
      log "$consec cells in a row failed; stopping (check the replicas, logs in $LOGS)"; break
    fi
  fi
done
log "budget $RUN: $ran ran, $done_ skipped as done, $over skipped for budget, $failed failed;" \
    "$(( SECONDS / 60 )) of $BUDGET_MIN min used"
(( failed == 0 ))
