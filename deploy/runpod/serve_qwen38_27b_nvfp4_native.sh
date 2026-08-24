#!/usr/bin/env bash
# M0-part2: same model/GPU as M0, but the NATIVE Blackwell sm_120 FP4 path.
# Requires the cu129 stack: runpod/pytorch:1.1.0-cu1290-torch291-ubuntu2404 (nvcc 12.9 so
# FlashInfer can JIT sm_120 kernels). Recipe re-validated 2026-08-24:
# https://recipes.vllm.ai/Qwen/Qwen3.8-27B?variant=nvfp4  (section "1x RTX 5090")
#
# Differences vs serve_qwen38_27b_nvfp4.sh (M0):
#   --linear-backend auto      expect log line "FlashInferCutlassNvFp4LinearKernel for NVFP4 GEMM"
#                              (M0 forced marlin: dequant->BF16 fallback)
#   no --attention-backend     let vLLM pick FlashInfer (M0 forced TRITON_ATTN)
#   no VLLM_USE_FLASHINFER_SAMPLER=0  nvcc 12.9 can build the sampler now
#   --enforce-eager KEPT       recipe confirms CUDA graph capture OOMs on 1x5090 (784 MiB alloc)
#                              regardless of --gpu-memory-utilization; not a toolchain issue
#   --mm-encoder-tp-mode data  recipe default (Qwen3.8 is multimodal; text-only serving verified)
# Run rows: #1 A/B vs M0 -> plain `bash` (kernels are the ONLY change vs M0);
#           #2 recipe-recommended MTP -> `MTP=1 bash` (spec decoding, method "mtp").
# Other levers (recipe, KV pool at 32K ctx): --language-model-only -> 135,926 tokens;
#   + --max-num-seqs 8 -> 152,917. Pass via EXTRA_ARGS.
set -euo pipefail
export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# vllm 0.26.x kernels link libcudart.so.13, shipped as the nvidia-cuda-runtime pip dep (torch itself
# is cu129); the loader needs the path spelled out or `import vllm` dies on libcudart.so.13.
export LD_LIBRARY_PATH="/workspace/.venv/lib/python3.12/site-packages/nvidia/cu13/lib:${LD_LIBRARY_PATH:-}"

MODEL="${MODEL:-Inferact/Qwen3.8-27B-NVFP4}"
PORT="${PORT:-8000}"
MAX_LEN="${MAX_LEN:-32768}"        # M0 A/B row; do a second run at 262144 for the KV story
GPU_UTIL="${GPU_UTIL:-0.95}"  # 0.92 leaves 1.65 GiB KV < 1.87 GiB floor once MTP head + vision tower load; safe in eager (no graph capture outside the budget)
LINEAR_BACKEND="${LINEAR_BACKEND:-auto}"
MTP="${MTP:-0}"                    # 1 -> MTP-3 spec decoding (0.754 acceptance per recipe)
EXTRA_ARGS="${EXTRA_ARGS:-}"

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
  --mm-encoder-tp-mode data \
  --linear-backend "$LINEAR_BACKEND" \
  --served-model-name qwen3.8-27b-nvfp4 \
  $([ "$MTP" = 1 ] && echo --speculative-config '{"method":"mtp","num_speculative_tokens":3}') \
  $EXTRA_ARGS
