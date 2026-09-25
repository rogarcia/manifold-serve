#!/usr/bin/env bash
# Follow-up to run-deferred.sh:
#   - TTFT test (ttft-test.py) on baseline and MTP k=5, FlashInfer (default) and TRITON_ATTN
#   - walkthrough §6 item 4: TRITON_ATTN vs FlashInfer. The startup log says FlashInfer lacks
#     fused multi-step draft decode; TRITON_ATTN may not. Baseline runs on TRITON_ATTN too, since
#     the flag changes the target model's attention as well, not only the draft path.
# Results: $M0/results/$RUN/<config>[-triton]/...
set -uo pipefail
source "$(dirname "$0")/env.sh"
S=$(dirname "$0")
export RUN=${RUN:-2026-09-24}
mkdir -p "$M0/results/$RUN"
rm -f "$M0/DONE"
cell() { "$S/bench.sh" "$@" || echo "FAILED cell: $CFG/$SUFFIX $*"; }

for spec in "baseline|" "mtp 5|" "baseline|triton" "mtp 5|triton"; do
  CFG=${spec%%|*}; SUFFIX=${spec##*|}
  if [ "$SUFFIX" = triton ]; then
    export CONFIG_SUFFIX=triton EXTRA_ARGS="--attention-backend TRITON_ATTN"
  else
    unset CONFIG_SUFFIX EXTRA_ARGS
  fi
  "$S/serve.sh" $CFG || { echo "FAILED to start: $CFG $SUFFIX"; continue; }
  echo "=== $(cat "$M0/current-config") $(date -u +%H:%M:%SZ)"
  python "$S/ttft-test.py" "$M0/results/$RUN/ttft-test.jsonl" || echo "FAILED ttft-test: $CFG $SUFFIX"
  if [ "$SUFFIX" = triton ]; then
    SEED=1 NUM_PROMPTS=80  cell chat 1
    SEED=1 NUM_PROMPTS=80  cell code 1
    SEED=8 NUM_PROMPTS=160 cell code 8
  fi
done
"$S/stop.sh"
date -u > "$M0/DONE"
