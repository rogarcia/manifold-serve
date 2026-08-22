#!/usr/bin/env bash
# M0: single vLLM replica of Qwen3.8-27B (NVFP4) on one RTX 5090 (32 GB, Blackwell sm_120).
# Source: https://recipes.vllm.ai/Qwen/Qwen3.8-27B?hardware=rtx_5090&variant=nvfp4
#
# Why each flag (see .scratch/m0-runpod-walkthrough.md for the long version):
#   --kv-cache-dtype fp8   KV cache in 8-bit -> ~2x more tokens of context fit in the same VRAM
#   --enforce-eager        skip CUDA-graph capture; on a 32 GB card capture OOMs at startup (recipe note)
#   --max-model-len 32768  the context ceiling that fits next to ~18 GB of NVFP4 weights with headroom
#   --reasoning-parser     Qwen3.8 is a thinking model; parser splits <think> from content in the API
#   --enable-prefix-caching  ON so multi-turn sessions in the loadgen actually exercise the cache
set -euo pipefail
export HF_HOME="${HF_HOME:-/workspace/hf}"                 # persist weights on the pod volume
export HF_HUB_ENABLE_HF_TRANSFER=1                           # faster 18 GB download
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True     # avoids fragmentation OOMs on 32 GB
export VLLM_USE_FLASHINFER_SAMPLER=0                         # image may lack nvcc; native sampler is fine

MODEL="${MODEL:-Inferact/Qwen3.8-27B-NVFP4}"
PORT="${PORT:-8000}"
MAX_LEN="${MAX_LEN:-32768}"
GPU_UTIL="${GPU_UTIL:-0.92}"

exec /workspace/.venv/bin/vllm serve "$MODEL" \
  --host 0.0.0.0 --port "$PORT" \
  --tensor-parallel-size 1 \
  --max-model-len "$MAX_LEN" \
  --gpu-memory-utilization "$GPU_UTIL" \
  --kv-cache-dtype fp8 \
  --enforce-eager \
  --enable-prefix-caching \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder \
  --served-model-name qwen3.8-27b-nvfp4
