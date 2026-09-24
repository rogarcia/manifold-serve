#!/usr/bin/env bash
# Start the vLLM server in tmux session "vllm" and wait until it is ready.
# Usage: serve.sh baseline | serve.sh mtp [K]      (K = num_speculative_tokens, default 5)
# The only difference between configs is --speculative-config.
set -euo pipefail
source "$(dirname "$0")/env.sh"

CONFIG=${1:?baseline|mtp}
K=${2:-5}
case "$CONFIG" in
  baseline) SPEC_ARGS=""; TAG=baseline ;;
  mtp)      SPEC_ARGS="--speculative-config '{\"method\":\"mtp\",\"num_speculative_tokens\":$K}'"; TAG="mtp-k$K" ;;
  *) echo "unknown config: $CONFIG" >&2; exit 2 ;;
esac

"$(dirname "$0")/stop.sh"
mkdir -p "$M0/logs"
LOG="$M0/logs/serve-$TAG-$(date -u +%Y%m%dT%H%M%SZ).log"
echo "$TAG" > "$M0/current-config"

# Recipe command verbatim (https://recipes.vllm.ai/Qwen/Qwen3.8-27B?variant=nvfp4_nvidia), plus --seed and --port.
tmux new-session -d -s vllm "source $(dirname "$0")/env.sh; vllm serve $MODEL --revision $MODEL_REVISION \
  --kv-cache-dtype fp8 \
  --max-model-len $MAX_MODEL_LEN \
  --max-num-seqs 8 \
  --max-num-batched-tokens 8192 \
  --enable-chunked-prefill \
  --async-scheduling \
  --enable-prefix-caching \
  --tensor-parallel-size 1 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_xml \
  --reasoning-parser qwen3 \
  --mm-encoder-tp-mode data \
  --seed 0 --port $PORT \
  $SPEC_ARGS 2>&1 | tee $LOG"

echo "log: $LOG"
for _ in $(seq 1 180); do
  if curl -sf "localhost:$PORT/health" >/dev/null; then echo "ready: $TAG"; exit 0; fi
  tmux has-session -t vllm 2>/dev/null || { echo "server exited; tail of log:" >&2; tail -40 "$LOG" >&2; exit 1; }
  sleep 5
done
echo "timed out after 15 min; tail of log:" >&2; tail -40 "$LOG" >&2; exit 1
