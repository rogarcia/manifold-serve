#!/usr/bin/env bash
# Exec one engine for slot SLOT, real vLLM on GPU SLOT or manifold.sim with SIM=1. It must exec:
# ProcessFleet and reclaim.sh send SIGTERM to this pid alone, and that has to reach the engine.
# The pad makes every start cost what a first start in a fresh container costs, so a one-container
# run does not measure the warm page cache. The slot is STARTING meanwhile, so P3 counts it as
# GPU-seconds held.
# Usage: engine.sh SLOT     SIM_START_S (1): the sim's --start-delay-s. SIM_FLAGS are added last.
set -euo pipefail
SLOT=${1:?usage: engine.sh SLOT}
source "$(dirname "$0")/common.sh"
PORT=${SLOT_PORTS[$SLOT]}
sleep "$(cat "$PAD_FILE" 2>/dev/null || echo "${START_PAD_S:-0}")"
if [ "$SIM" = 1 ]; then
  read -ra sim_flags <<< "${SIM_FLAGS:-}"
  exec python -m manifold.sim --host 127.0.0.1 --port "$PORT" --model "$MODEL" \
    --block-tokens "$SIM_BLOCK_TOKENS" --kv-blocks "$SIM_KV_BLOCKS" --seed 0 \
    --start-delay-s "${SIM_START_S:-1}" --decode-ms-per-token 11.5 --decode-ms-per-running 0.21 \
    --max-num-seqs 512 --shutdown-timeout-s 25 ${sim_flags[@]+"${sim_flags[@]}"}
fi
export CUDA_VISIBLE_DEVICES=$SLOT
vllm_argv "$PORT"
exec "${VLLM_ARGV[@]}" --shutdown-timeout 25
