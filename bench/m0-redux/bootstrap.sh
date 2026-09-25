#!/usr/bin/env bash
# Run once on a fresh RunPod pod to prepare an M0-redux run. Idempotent.
# Image used: runpod/pytorch:1.0.7-cu1300-torch291-ubuntu2404-cluster (nvcc 13.0, driver 580.x).
# Installs uv + the pinned vLLM stack into /workspace/.venv (persistent volume), downloads the
# model and datasets into /workspace/hf, and writes the environment manifest.
#
# HF token (optional, raises download rate limits): put HF_TOKEN=hf_... in
# /workspace/m0-redux/.env (see .env.example), or export HF_TOKEN before running.
# The token is only read from there; it is never written to logs or the manifest.
set -euo pipefail
source "$(dirname "$0")/env.sh"
cd "$WORKSPACE"
mkdir -p "$M0"/{logs,results,data}

# torch's CUDA major must match the image's nvcc, which FlashInfer uses to JIT sm_120 kernels.
nvcc --version | grep -E 'release 13\.' \
  || { echo "FATAL: expected a CUDA 13.x toolkit for the cu130 torch build"; nvcc --version; exit 1; }

command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
[ -d "$VENV" ] || uv venv "$VENV" --python 3.12

# On network-backed volumes (RunPod network volumes) uv's temp-file-then-rename writes can fail
# with "Stale file handle (os error 116)". Packages installed before the failure stay installed,
# so retrying makes progress. The 2026-09-21 and 09-24 pods did not hit this.
echo "/workspace filesystem: $(findmnt -no FSTYPE,SOURCE -T "$WORKSPACE")"
retry() {
  local n
  for n in 1 2 3 4 5; do
    "$@" && return 0
    echo "attempt $n failed: $*" >&2; sleep 5
  done
  return 1
}

# Versions pinned to the 2026-09-21 run. FROZEN=1 installs the exact recorded freeze instead.
if [ "${FROZEN:-0}" = 1 ]; then
  retry uv pip install --python "$VENV/bin/python" --torch-backend=cu130 -r "$(dirname "$0")/results/pip-freeze.txt"
else
  # [bench] pulls `datasets` etc. for `vllm bench serve`; transformers >= 5.8.0 is a recipe prerequisite.
  retry uv pip install --python "$VENV/bin/python" --torch-backend=cu130 "vllm[bench]==0.29.0" "transformers==5.17.0"
fi

if [ -n "${HF_TOKEN:-}" ] || [ -s "$HF_HOME/token" ]; then echo "HF token: set"; else echo "HF token: not set (anonymous rate limits)"; fi
retry hf download "$MODEL" --revision "$MODEL_REVISION" >/dev/null
retry hf download --repo-type dataset likaixin/InstructCoder >/dev/null
retry hf download --repo-type dataset philschmid/mt-bench >/dev/null
[ -f "$M0/data/spec_bench.jsonl" ] || curl -sfL -o "$M0/data/spec_bench.jsonl" \
  https://raw.githubusercontent.com/hemingkx/Spec-Bench/refs/heads/main/data/spec_bench/question.jsonl

{ date -u; uname -r
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
  nvcc --version | grep release
  python -c "import torch,transformers,vllm;print('vllm',vllm.__version__,'torch',torch.__version__,'transformers',transformers.__version__)" 2>&1 | tail -1
  echo "model $MODEL @ $(cat "$HF_HOME/hub/models--${MODEL//\//--}/refs/main" 2>/dev/null || echo "$MODEL_REVISION")"
} | tee "$M0/env-manifest.txt"
uv pip freeze --python "$VENV/bin/python" > "$M0/pip-freeze.txt"
echo "bootstrap done. Next: tmux new -d -s matrix '$M0/scripts/run-code.sh 2>&1 | tee $M0/logs/run-code.log'"
