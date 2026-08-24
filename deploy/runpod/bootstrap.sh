#!/usr/bin/env bash
# Run once on a fresh RunPod pod (runpod/pytorch:* image). Installs uv + vLLM into a
# venv on the persistent /workspace volume so it survives stop/start.
# M0-part2 image: runpod/pytorch:1.1.0-cu1290-torch291-ubuntu2404 (nvcc 12.9 -> sm_120 JIT works).
set -euo pipefail
cd /workspace
# Fail fast if the toolchain can't JIT sm_120 FlashInfer kernels (M0's root cause was nvcc 12.8).
nvcc --version | grep -E 'release (12\.(9|[1-9][0-9])|1[3-9]\.)' \
  || { echo "FATAL: nvcc >= 12.9 required for native sm_120 kernels"; nvcc --version; exit 1; }
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
[ -d .venv ] || uv venv .venv --python 3.12
# --torch-backend=cu129, NOT auto: auto matches the DRIVER (CUDA 13.0 here -> pulled cu132 torch),
# but FlashInfer JIT compiles with the image's nvcc 12.9, and CUDA compat only holds within a major
# version. Torch's CUDA must match nvcc. Bump this only in lockstep with the image's toolkit.
# vllm==0.26.*: the 0.27.x PyPI wheels are CUDA-13-linked (libcudart.so.13) with no cu129 variant
# published; 0.26.x pairs with torch cu129 and is the line the recipe's 5090 section was verified on.
# Its kernels still link libcudart.so.13, shipped via the nvidia-cuda-runtime pip dep — the serve
# script must put site-packages/nvidia/cu13/lib on LD_LIBRARY_PATH or `import vllm` fails.
# transformers >= 5.8.0: recipe prerequisite (Qwen3-VL processor classes for Qwen3.8's config.json).
# ninja/cmake: FlashInfer JIT-compiles the NVFP4 cutlass kernel at startup and shells out to ninja
# (first failure mode seen: FileNotFoundError 'ninja' mid-profile-run).
uv pip install --python .venv/bin/python --torch-backend=cu129 "vllm==0.26.*" "transformers>=5.8.0" ninja cmake
LD_LIBRARY_PATH="/workspace/.venv/lib/python3.12/site-packages/nvidia/cu13/lib:${LD_LIBRARY_PATH:-}" \
.venv/bin/vllm --version
.venv/bin/python -c "import torch,transformers;print('torch',torch.__version__,'cuda',torch.version.cuda,'transformers',transformers.__version__)"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
