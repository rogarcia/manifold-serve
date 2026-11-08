#!/usr/bin/env bash
# The probe, then cells C1 to C7 in priority order. A step the deadline cannot fit is skipped and a
# later, shorter one may still run. Cells with DONE are skipped, so rerunning resumes. The probe is
# reused only when $M2/probe.env was written for this RUN.
# Usage: BUDGET_MIN=88 plan.sh RUN   |   plan.sh --cost
#   --cost prints the plan and exits 1 when the required steps exceed MAX_USD (15.63) or MAX_MIN (95).
#     C7 is optional: the spec names it first to drop, and the deadline drops it.
#   USD_PER_HOUR (7.33), MAX_CONSEC_FAIL (2), TIME_SCALE (1, also divides the estimates).
set -euo pipefail
USD_PER_HOUR=${USD_PER_HOUR:-7.33} MAX_USD=${MAX_USD:-15.63} MAX_MIN=${MAX_MIN:-95}

# C1 and C6 exceed the spec's 19 minutes: the second cold start begins at t=960, so at S=170 the
# slot is READY at t=1130 and its 360 held requests finish after t=1160. C3 and C4 run over because
# gate-off overload leaves a backlog at the end of the window. Columns: name mode seed gate min use.
steps() { cat <<'EOF'
probe          -         - -   8  required
c1-reconcile-s0 reconcile 0 on  20 required
c2-static-2-s0  static-2  0 on  10 required
c3-static-1-blind-s0 static-1 0 off 12 required
c4-static-2-blind-s0 static-2 0 off 11 required
c5-reclaim      reclaim   0 on  7  required
c6-reconcile-s1 reconcile 1 on  20 required
c7-static-2-s1  static-2  1 on  10 optional
EOF
}

if [ "${1:-}" = --cost ]; then
  steps | awk -v rate="$USD_PER_HOUR" -v max_usd="$MAX_USD" -v max_min="$MAX_MIN" '
    BEGIN { printf "%-22s %-10s %5s %4s %8s %s\n", "step", "fleet", "gate", "min", "usd", "use" }
    { usd = $5 / 60 * rate; printf "%-22s %-10s %5s %4d %8.2f %s\n", $1, $2, $4, $5, usd, $6
      all += $5; if ($6 == "required") req += $5 }
    END {
      printf "required %d min, $%.2f; with optional %d min, $%.2f; limits %d min, $%.2f\n",
        req, req / 60 * rate, all, all / 60 * rate, max_min, max_usd
      if (all > max_min || all / 60 * rate > max_usd) print "the optional steps do not fit and the deadline will skip them"
      exit (req > max_min || req / 60 * rate > max_usd) ? 1 : 0 }'
  exit
fi

RUN=${1:?usage: BUDGET_MIN=minutes plan.sh RUN | plan.sh --cost}
[[ "${BUDGET_MIN:-}" =~ ^[0-9]+$ ]] || { sed -n '5,8p' "$0" >&2; exit 2; }
source "$(dirname "$0")/common.sh"
export RUN
TIME_SCALE=${TIME_SCALE:-1}
export TIME_SCALE
deadline=$(( SECONDS + BUDGET_MIN * 60 ))
RESULTS_DIR=$M2/results/$RUN
mkdir -p "$RESULTS_DIR"
trap stop_all EXIT

probe_done() { [ -f "$M2/probe.env" ] && grep -qx "PROBE_RUN=$RUN" "$M2/probe.env"; }
estimate_s() { awk -v m="$1" -v k="$TIME_SCALE" 'BEGIN { printf "%d\n", m * 60 / k }'; }

ran=0 done_=0 over=0 failed=0 consec=0
while read -r name mode seed gate min _; do
  left=$(( deadline - SECONDS ))
  if [ "$name" = probe ]; then
    if probe_done; then log "skip (done): probe"; continue; fi
    "$M2_DIR/probe.sh" </dev/null || die "probe failed: no cell can be trusted"
    cp "$M2/probe.env" "$RESULTS_DIR/probe.env"
    continue
  fi
  if [ -f "$RESULTS_DIR/$name/DONE" ]; then
    log "skip (DONE): $name"; done_=$(( done_ + 1 )); continue
  fi
  cost=$(estimate_s "$min")
  if (( cost > left )); then
    log "skip (budget): $name needs ${cost}s, ${left}s left"; over=$(( over + 1 )); continue
  fi
  log "$name ($left s left)"
  rc=0
  GATE=$gate "$M2_DIR/cell.sh" "$name" "$mode" "$seed" </dev/null || rc=$?
  if (( rc == 0 )); then
    ran=$(( ran + 1 )); consec=0
  else
    failed=$(( failed + 1 )); consec=$(( consec + 1 ))
    if (( consec >= ${MAX_CONSEC_FAIL:-2} )); then
      log "$consec cells in a row failed; stopping"; break
    fi
  fi
done < <(steps)

log "plan $RUN: $ran ran, $done_ skipped as done, $over skipped for budget, $failed failed;" \
    "$(( SECONDS / 60 )) of $BUDGET_MIN min used"
python "$M2_DIR/summarize.py" "$RESULTS_DIR" | tee "$RESULTS_DIR/summary.txt" || log "summarize.py failed"
(( failed == 0 ))
