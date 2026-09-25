#!/usr/bin/env bash
# Walkthrough §6 item 7: open-loop Poisson rate sweep with the M0 loadgen (manifold/loadgen),
# same protocol as bench/m0/sweep.sh (32 multi-turn sessions, 120 s per rate, max_tokens 120,
# thinking off, 45 s warmup), on baseline and MTP k=5 (FlashInfer, recipe default).
# The loadgen must be the version that counts tokens from usage, not SSE chunks.
# Needs the repo's manifold/ package at $M0/manifold (push it with scp -r).
# Results: $M0/results/$RUN/sweep/<config>/{rate_R.csv,summary_R.txt,metrics_{before,after}_R.txt}
set -uo pipefail
source "$(dirname "$0")/env.sh"
S=$(dirname "$0")
export RUN=${RUN:-2026-09-24}
export PYTHONPATH="$M0${PYTHONPATH:+:$PYTHONPATH}"
RATES=${RATES:-1 2 4 6 8 10}
DUR=${DUR:-120}
rm -f "$M0/DONE"
unset CONFIG_SUFFIX EXTRA_ARGS
BASE="http://localhost:$PORT"
metrics() { curl -s "$BASE/metrics" | grep -E '^vllm:(prefix_cache_(hits|queries)_total|num_requests_(waiting|running)|kv_cache_usage_perc|gpu_cache_usage_perc|spec_decode_num_(drafts|draft_tokens|accepted_tokens)_total|generation_tokens_total|prompt_tokens_total)' > "$1"; }

for CFG in "baseline" "mtp 5"; do
  "$S/serve.sh" $CFG || { echo "FAILED to start: $CFG"; continue; }
  TAG=$(cat "$M0/current-config"); OUT=$M0/results/$RUN/sweep/$TAG; mkdir -p "$OUT"
  echo "=== $TAG $(date -u +%H:%M:%SZ) warmup"
  python -m manifold.loadgen --base-url "$BASE" --model "$MODEL" --rate 1 --duration 45 --sessions 16 --disable-thinking >/dev/null \
    || echo "FAILED warmup: $TAG"
  for r in $RATES; do
    echo "=== $TAG rate $r req/s $(date -u +%H:%M:%SZ)"
    metrics "$OUT/metrics_before_$r.txt"
    python -m manifold.loadgen --base-url "$BASE" --model "$MODEL" --rate "$r" --duration "$DUR" \
      --sessions 32 --disable-thinking --out "$OUT/rate_$r.csv" | tee "$OUT/summary_$r.txt" \
      || echo "FAILED rate: $TAG $r"
    metrics "$OUT/metrics_after_$r.txt"
  done
done
"$S/stop.sh"
date -u > "$M0/DONE"
