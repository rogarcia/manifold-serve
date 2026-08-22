#!/usr/bin/env bash
# M0 load sweep: open-loop Poisson arrivals at increasing rates against one vLLM replica.
# Run ON the pod (no network noise): cd /workspace/manifold-serve && bash bench/m0/sweep.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
BASE="${BASE:-http://localhost:8000}"
MODEL="${MODEL:-qwen3.8-27b-nvfp4}"
DUR="${DUR:-120}"
OUT="${OUT:-bench/m0}"
mkdir -p "$OUT"
echo "warmup"; uv run python -m manifold.loadgen --base-url "$BASE" --model "$MODEL" --rate 1 --duration 45 --sessions 16 --disable-thinking >/dev/null
for r in ${RATES:-0.5 1 2 4 8}; do
  echo "=== rate $r req/s ==="
  curl -s "$BASE/metrics" | grep -E '^vllm:(prefix_cache_(hits|queries)_total|num_requests_waiting|gpu_cache_usage_perc)' > "$OUT/metrics_before_$r.txt"
  uv run python -m manifold.loadgen --base-url "$BASE" --model "$MODEL" --rate "$r" --duration "$DUR" \
      --sessions 32 --disable-thinking --out "$OUT/rate_$r.csv" | tee "$OUT/summary_$r.txt"
  curl -s "$BASE/metrics" | grep -E '^vllm:(prefix_cache_(hits|queries)_total|num_requests_waiting|gpu_cache_usage_perc)' > "$OUT/metrics_after_$r.txt"
done
