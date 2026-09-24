#!/usr/bin/env bash
# Code-only A/B: baseline vs MTP k=1,2,3,5 on InstructCoder, concurrency 1 and 8, greedy, thinking off.
# One server start per config. Each concurrency uses its own seed so the prompts differ
# within a server (no prefix-cache hits) while staying identical across configs.
set -uo pipefail
source "$(dirname "$0")/env.sh"
S=$(dirname "$0")
rm -f "$M0/DONE"
for cfg in "baseline" "mtp 5" "mtp 3" "mtp 2" "mtp 1"; do
  $S/serve.sh $cfg || { echo "FAILED to start: $cfg"; continue; }
  SEED=1 NUM_PROMPTS=80  $S/bench.sh code 1 || echo "FAILED cell: $cfg code c1"
  SEED=8 NUM_PROMPTS=160 $S/bench.sh code 8 || echo "FAILED cell: $cfg code c8"
done
$S/stop.sh
date -u > "$M0/DONE"
