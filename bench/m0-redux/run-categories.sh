#!/usr/bin/env bash
# Walkthrough §6 item 6, part 1: Spec-Bench categories (summarization, translation, RAG, QA)
# on baseline and MTP k=5, FlashInfer (recipe default), c=1, 40 prompts, greedy, thinking off.
# Results: $M0/results/$RUN/<config>/...
set -uo pipefail
source "$(dirname "$0")/env.sh"
S=$(dirname "$0")
export RUN=${RUN:-2026-09-24}
rm -f "$M0/DONE"
cell() { "$S/bench.sh" "$@" || echo "FAILED cell: $CFG $*"; }
unset CONFIG_SUFFIX EXTRA_ARGS
for CFG in "baseline" "mtp 5"; do
  "$S/serve.sh" $CFG || { echo "FAILED to start: $CFG"; continue; }
  echo "=== $(cat "$M0/current-config") $(date -u +%H:%M:%SZ)"
  for c in summarize translate rag qa; do SEED=1 NUM_PROMPTS=40 cell $c 1; done
done
"$S/stop.sh"
date -u > "$M0/DONE"
