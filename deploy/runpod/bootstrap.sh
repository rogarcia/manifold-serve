#!/usr/bin/env bash
# Run once on a fresh RunPod pod (runpod/pytorch:*-cu1281-* image). Installs uv + vLLM into a
# venv on the persistent /workspace volume so it survives stop/start.
set -euo pipefail
cd /workspace
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
[ -d .venv ] || uv venv .venv --python 3.12
# vLLM wheels pull the matching torch (cu128, needed for Blackwell sm_120).
uv pip install --python .venv/bin/python "vllm>=0.17"
.venv/bin/vllm --version
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
