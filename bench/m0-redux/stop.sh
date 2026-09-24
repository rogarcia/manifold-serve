#!/usr/bin/env bash
# Stop the vLLM server and wait for the GPU to be released.
source "$(dirname "$0")/env.sh"
tmux kill-session -t vllm 2>/dev/null || true
pkill -f "bin/vllm serve" 2>/dev/null || true
for _ in $(seq 1 30); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
  [ "$used" -lt 1000 ] && exit 0
  sleep 2
done
echo "GPU still holds ${used} MiB after 60s" >&2; exit 1
