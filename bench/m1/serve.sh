#!/usr/bin/env bash
# Start replica A (GPU 0, PORT_A), wait for /health, then replica B (GPU 1, PORT_B), in tmux
# sessions vllm-a and vllm-b. Writes $M1/current.env describing what is running.
# Usage: serve.sh MODEL_TAG [PRESSURE]     PRESSURE: nocap (default) | fits | pressure
#   nocap leaves the KV pool to vLLM (bring-up, calibration). fits and pressure pass
#   --kv-cache-memory-bytes from $M1/pressure-<tag>.env (written by size_kv.py).
# SIM=1 starts `python -m manifold.sim` instead, sized in blocks from the same file.
# Everything binds to 127.0.0.1: VLLM_SERVER_DEV_MODE exposes /reset_prefix_cache and other dev
# endpoints, and the load generator runs on the same pod.
set -euo pipefail
export MODEL_TAG=${1:?usage: serve.sh MODEL_TAG [nocap|fits|pressure]}
PRESSURE=${2:-nocap}
source "$(dirname "$0")/env.sh"
source "$BENCH_DIR/lib.sh"

KV_BYTES=""; KV_SIM_BLOCKS=$SIM_KV_BLOCKS
case "$PRESSURE" in
  nocap) ;;
  fits|pressure)
    v=KV_BYTES_$PRESSURE; t=KV_TOKENS_$PRESSURE
    if [ "$SIM" = 1 ]; then
      [ -n "${!t:-}" ] || die "$t not set: run size_kv.py for $MODEL_TAG first"
      KV_SIM_BLOCKS=$(( ${!t} * SIM_BYTES_PER_TOKEN / SIM_BLOCK_TOKENS ))
    else
      [ -n "${!v:-}" ] || die "$v not set: run size_kv.py for $MODEL_TAG first"
      KV_BYTES=${!v}
    fi ;;
  *) die "unknown pressure: $PRESSURE (nocap | fits | pressure)" ;;
esac

"$BENCH_DIR/stop.sh"
rm -f "$M1/current.env"
mkdir -p "$LOGS"
TS=$(date -u +%Y%m%dT%H%M%SZ)

start_replica() {
  local r=$1 gpu=$2 port=$3 logf=$4
  if [ "$SIM" = 1 ]; then
    # shellcheck disable=SC2086
    launch "vllm-$r" "$logf" python -m manifold.sim --host 127.0.0.1 --port "$port" --model "$MODEL" \
      --block-tokens "$SIM_BLOCK_TOKENS" --kv-blocks "$KV_SIM_BLOCKS" --seed 0 $SIM_FLAGS
  else
    # shellcheck disable=SC2086
    launch "vllm-$r" "$logf" env CUDA_VISIBLE_DEVICES="$gpu" \
      vllm serve "$MODEL" --revision "$MODEL_REVISION" --host 127.0.0.1 --port "$port" \
      --enable-prefix-caching \
      --max-model-len "$MAX_MODEL_LEN" \
      --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" \
      --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
      ${KV_BYTES:+--kv-cache-memory-bytes "$KV_BYTES"} \
      --seed 0 \
      --default-chat-template-kwargs "$CHAT_TEMPLATE_KWARGS" \
      $MODEL_FLAGS
  fi
  log "replica $r: log $logf"
  wait_http "http://localhost:$port/health" 1200 "vllm-$r" "$logf"
  log "replica $r ready"
}

LOG_A=$LOGS/serve-$MODEL_TAG-$PRESSURE-a-$TS.log
LOG_B=$LOGS/serve-$MODEL_TAG-$PRESSURE-b-$TS.log
# One after the other: vLLM profiles free memory at startup.
start_replica a 0 "$PORT_A" "$LOG_A"
start_replica b 1 "$PORT_B" "$LOG_B"

eval "$(parse_serve_log "$LOG_A" | sed 's/^/A_/')"
eval "$(parse_serve_log "$LOG_B" | sed 's/^/B_/')"
# shellcheck disable=SC2154
{
  [ "$A_BLOCK_TOKENS" = "$B_BLOCK_TOKENS" ] || die "block size differs: A $A_BLOCK_TOKENS, B $B_BLOCK_TOKENS"
  [ "$A_CUDAGRAPH_SIZES" = "$B_CUDAGRAPH_SIZES" ] || die "cudagraph capture sizes differ: A $A_CUDAGRAPH_SIZES, B $B_CUDAGRAPH_SIZES"
  [ -n "$A_KV_TOKENS" ] && [ -n "$B_KV_TOKENS" ] || die "no 'GPU KV cache size' line in the startup logs"
  cat > "$M1/current.env" <<EOF
CUR_MODEL_TAG=$MODEL_TAG
CUR_PRESSURE=$PRESSURE
CUR_KV_BYTES=$KV_BYTES
CUR_SIM=$SIM
CUR_BLOCK_TOKENS=$A_BLOCK_TOKENS
CUR_KV_TOKENS_A=$A_KV_TOKENS
CUR_KV_TOKENS_B=$B_KV_TOKENS
CUR_SERVE_LOG_A=$LOG_A
CUR_SERVE_LOG_B=$LOG_B
EOF
}
log "serving $MODEL_TAG ($PRESSURE): block $A_BLOCK_TOKENS tokens, KV pool A $A_KV_TOKENS / B $B_KV_TOKENS tokens"
