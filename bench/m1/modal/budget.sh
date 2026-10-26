#!/usr/bin/env bash
# Run locally: start bench/m1/budget.sh on Modal, detached, with a hard spending cap. CAP_MIN caps
# each container's run time (the Function timeout). The budget.sh deadline is fixed in wall-clock
# time when this script runs, so a preempted call that restarts in a new container keeps the
# original deadline instead of starting a fresh budget.
# Usage: bench/m1/modal/budget.sh RUN CAP_MIN      e.g. bench/m1/modal/budget.sh m1-2026-10-26 115
#   STOP_MIN (5): minutes kept back from the deadline for stop.sh and the Volume commit.
#   PLAN_FILE: passed to budget.sh, relative to bench/m1 (e.g. plans/pressure-recheck.txt).
set -euo pipefail
RUN=${1:?usage: budget.sh RUN CAP_MIN}
CAP_MIN=${2:?usage: budget.sh RUN CAP_MIN}
[[ "$CAP_MIN" =~ ^[0-9]+$ ]] || { echo "CAP_MIN must be whole minutes" >&2; exit 2; }
D=$(cd "$(dirname "$0")" && pwd)
deadline=$(( $(date +%s) + (CAP_MIN - ${STOP_MIN:-5}) * 60 ))
echo "cap ${CAP_MIN} min per container; budget.sh deadline $(date -r "$deadline" '+%F %T %Z')"
cmd="${PLAN_FILE:+PLAN_FILE=$PLAN_FILE }BUDGET_MIN=\$(( ($deadline - \$(date +%s)) / 60 )) ./budget.sh $RUN 2>&1 | tee -a /workspace/m1/logs/budget.log"
M1_TIMEOUT_S=$(( CAP_MIN * 60 )) uv run --no-sync modal run --detach "$D/app.py" --cmd "$cmd"
