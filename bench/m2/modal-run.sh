#!/usr/bin/env bash
# Run locally: start bench/m2/plan.sh on Modal, detached, with a hard spending cap. CAP_MIN caps the
# container's run time (the Function timeout). The deadline is fixed in wall-clock time here, so a
# preempted call that restarts in a new container keeps it instead of starting a fresh budget.
# Usage: bench/m2/modal-run.sh RUN CAP_MIN      e.g. bench/m2/modal-run.sh m2-2026-11-08 95
#   STOP_MIN (5): minutes kept back from the deadline for shutdown and the Volume commit.
set -euo pipefail
RUN=${1:?usage: modal-run.sh RUN CAP_MIN}
CAP_MIN=${2:?usage: modal-run.sh RUN CAP_MIN}
[[ "$CAP_MIN" =~ ^[0-9]+$ ]] || { echo "CAP_MIN must be whole minutes" >&2; exit 2; }
D=$(cd "$(dirname "$0")" && pwd)
"$D/plan.sh" --cost
deadline=$(( $(date +%s) + (CAP_MIN - ${STOP_MIN:-5}) * 60 ))
echo "cap ${CAP_MIN} min per container; plan.sh deadline $(date -r "$deadline" '+%F %T %Z')"
cmd="BUDGET_MIN=\$(( ($deadline - \$(date +%s)) / 60 )) ../m2/plan.sh $RUN 2>&1 | tee -a /workspace/m1/logs/m2-plan.log"
M1_TIMEOUT_S=$(( CAP_MIN * 60 )) uv run --no-sync modal run --detach "$D/../m1/modal/app.py" --cmd "$cmd"
