#!/usr/bin/env bash
# Stop the router and both replicas, then wait for every GPU to be released.
set -uo pipefail
source "$(dirname "$0")/env.sh"
source "$BENCH_DIR/lib.sh"

"$BENCH_DIR/router.sh" stop
for s in vllm-a vllm-b; do tmux kill-session -t "$s" 2>/dev/null || true; done
pkill -f "bin/vllm serve" 2>/dev/null || true
pkill -f "manifold.sim" 2>/dev/null || true

if command -v nvidia-smi >/dev/null; then
  for _ in $(seq 1 45); do
    busy=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1 >= 1000' | wc -l)
    [ "$busy" -eq 0 ] && exit 0
    sleep 2
  done
  nvidia-smi --query-gpu=index,memory.used --format=csv >&2
  die "GPU memory still in use after 90 s"
fi
for _ in $(seq 1 15); do
  curl -sf --max-time 1 "$URL_A/health" >/dev/null || curl -sf --max-time 1 "$URL_B/health" >/dev/null || exit 0
  sleep 1
done
die "a replica still answers /health after 15 s"
