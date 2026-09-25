#!/usr/bin/env bash
# Deferred items from the 2026-09-21 run (walkthrough §6), on baseline and MTP k=5,7:
#   1. chat c1 repeat on a fresh server + chunk-timing probe (TTFT gap 170 vs 110 ms)
#   2. seed repeats: code c1 seeds 1,2,3 and code c8 seeds 8,9,10 (seed 1/8 = the 09-21 cells, the anchor)
#   3. sampling at Qwen non-thinking settings (T=0.7 top_p 0.8 top_k 20) on the same prompts
#   5. thinking on vs off, GSM8K c1, 40 prompts, output cap 4096
# Cells after the first on each server reuse prompts, so their TTFT includes prefix-cache hits;
# only the first chat cell and the code/math seed-new cells have clean TTFT.
# Results: $M0/results/$RUN/<config>/...
set -uo pipefail
source "$(dirname "$0")/env.sh"
S=$(dirname "$0")
export RUN=${RUN:-2026-09-24}
mkdir -p "$M0/results/$RUN"
rm -f "$M0/DONE"
cell() { "$S/bench.sh" "$@" || echo "FAILED cell: $CFG $*"; }

for CFG in "baseline" "mtp 5" "mtp 7"; do
  tag=$([ "$CFG" = baseline ] && echo baseline || echo "mtp-k${CFG#mtp }")
  # Reuse a server that is already up with this config (saves a restart on the first config).
  if [ "$(cat "$M0/current-config" 2>/dev/null)" = "$tag" ] && curl -sf "localhost:$PORT/health" >/dev/null; then
    echo "reusing running server: $tag"
  else
    "$S/serve.sh" $CFG || { echo "FAILED to start: $CFG"; continue; }
  fi
  echo "=== $tag $(date -u +%H:%M:%SZ)"
  SEED=1 NUM_PROMPTS=80 cell chat 1                                    # 1. first cell: clean TTFT
  python "$S/probe-ttft.py" "$M0/results/$RUN/ttft-probe.jsonl" 5 || echo "FAILED probe: $CFG"
  for s in 1 2 3;  do SEED=$s NUM_PROMPTS=80  cell code 1; done       # 2.
  for s in 8 9 10; do SEED=$s NUM_PROMPTS=160 cell code 8; done
  SEED=1 NUM_PROMPTS=80  cell code 1 qwen                              # 3.
  SEED=8 NUM_PROMPTS=160 cell code 8 qwen
  SEED=1 NUM_PROMPTS=80  cell chat 1 qwen
  SEED=1 NUM_PROMPTS=40  cell math 1 greedy off                        # 5. control
  SEED=1 NUM_PROMPTS=40 OUTPUT_LEN=4096 cell math 1 greedy on
done
"$S/stop.sh"
date -u > "$M0/DONE"
