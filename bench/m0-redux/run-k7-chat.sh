#!/usr/bin/env bash
# Follow-up: MTP k=7 on code, plus the chat cell (mt-bench) for baseline and MTP k=3,5,7.
# mt-bench has 80 prompts, so c1 and c8 share prompts within a server: c8 TTFT sees prefix-cache hits.
set -uo pipefail
source "$(dirname "$0")/env.sh"
S=$(dirname "$0")
rm -f "$M0/DONE"
for cfg in "mtp 7" "baseline" "mtp 5" "mtp 3"; do
  $S/serve.sh $cfg || { echo "FAILED to start: $cfg"; continue; }
  if [ "$cfg" = "mtp 7" ]; then
    SEED=1 NUM_PROMPTS=80  $S/bench.sh code 1 || echo "FAILED cell: $cfg code c1"
    SEED=8 NUM_PROMPTS=160 $S/bench.sh code 8 || echo "FAILED cell: $cfg code c8"
  fi
  SEED=1 NUM_PROMPTS=80 $S/bench.sh chat 1 || echo "FAILED cell: $cfg chat c1"
  SEED=1 NUM_PROMPTS=80 $S/bench.sh chat 8 || echo "FAILED cell: $cfg chat c8"
done
$S/stop.sh
date -u > "$M0/DONE"
