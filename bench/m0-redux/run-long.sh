#!/usr/bin/env bash
# Walkthrough §6 item 6, part 2: does MTP acceptance fall with context length?
# SPEED-Bench throughput buckets 1k/8k/16k/32k, category "mixed" only (fixed content mix across
# buckets), baseline vs MTP k=5, FlashInfer, c=1, 16 prompts, output cap 1024, greedy, thinking off.
# Needs $M0/data/speed_bench/throughput_{1k,8k,16k,32k}.jsonl (speed-bench-prepare.py).
set -uo pipefail
source "$(dirname "$0")/env.sh"
S=$(dirname "$0")
export RUN=${RUN:-2026-09-24} SPEED_CAT=mixed
rm -f "$M0/DONE"
cell() { "$S/bench.sh" "$@" || echo "FAILED cell: $CFG $*"; }
unset CONFIG_SUFFIX EXTRA_ARGS
for CFG in "baseline" "mtp 5"; do
  "$S/serve.sh" $CFG || { echo "FAILED to start: $CFG"; continue; }
  echo "=== $(cat "$M0/current-config") $(date -u +%H:%M:%SZ)"
  for b in long1k long8k long16k long32k; do SEED=1 NUM_PROMPTS=16 cell $b 1; done
done
"$S/stop.sh"
date -u > "$M0/DONE"
