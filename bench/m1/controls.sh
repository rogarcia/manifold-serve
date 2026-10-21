#!/usr/bin/env bash
# Controls under pressure at 70% of the knee, four policies, seed 1:
#   W0, ShareGPT first turns only, where prefix_aware must show no gain (a harness check).
#   W2, a ~6,000-token prefix shared by every conversation with short turns (w2-shared-prefix.json).
# Usage: controls.sh RUN
set -euo pipefail
RUN=${1:?usage: controls.sh RUN}
D=$(dirname "$0")
rc=0
WORKLOAD=w0 PRESSURES=pressure LOADS=70 SEEDS=${SEEDS:-1} "$D/matrix.sh" "$RUN" || rc=1
WORKLOAD=w2 PRESSURES=pressure LOADS=70 SEEDS=${SEEDS:-1} "$D/matrix.sh" "$RUN" || rc=1
exit "$rc"
