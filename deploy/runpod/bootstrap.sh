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
# --torch-backend=auto (recipe install): uv picks the torch CUDA-variant index matching the driver,
# so torch's CUDA (expect cu129 here) matches the nvcc that JIT-compiles FlashInfer. Without it,
# vLLM's default torch pin decides (M0 silently got cu128 torch this way).
# transformers >= 5.8.0: recipe prerequisite (Qwen3-VL processor classes for Qwen3.8's config.json).
uv pip install --python .venv/bin/python --torch-backend=auto "vllm>=0.17" "transformers>=5.8.0"
.venv/bin/vllm --version
.venv/bin/python -c "import torch,transformers;print('torch',torch.__version__,'cuda',torch.version.cuda,'transformers',transformers.__version__)"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
