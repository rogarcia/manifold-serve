#!/usr/bin/env bash
# Run once on a fresh pod (push.sh --bootstrap does it). Idempotent: rerun after any failure.
# Image: runpod/pytorch:1.0.7-cu1300-torch291-ubuntu2404-cluster (nvcc 13.0).
# Installs uv and the pinned stack into $VENV, downloads both models (resolving and recording
# their commit SHAs in $M1/revisions.env), ShareGPT and pg1184.txt, converts ShareGPT, and writes
# $M1/env-manifest.txt and $M1/pip-freeze.txt. FROZEN=1 installs bench/m1/results/manifest/pip-freeze.txt.
set -euo pipefail
source "$(dirname "$0")/env.sh"
source "$BENCH_DIR/lib.sh"
mkdir -p "$LOGS" "$RESULTS" "$DATA"
cd "$WORKSPACE"

# torch's CUDA major must match the image's nvcc, which FlashInfer uses to JIT sm_120 kernels.
nvcc --version | grep -E 'release 13\.' \
  || { echo "FATAL: expected a CUDA 13.x toolkit for the cu130 torch build"; nvcc --version; exit 1; }
command -v tmux >/dev/null || { apt-get update -qq && apt-get install -y -qq tmux; }
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
[ -d "$VENV" ] || uv venv "$VENV" --python 3.12

# On network-backed volumes uv's temp-file-then-rename writes can fail with "Stale file handle
# (os error 116)". Packages installed before the failure stay installed, so retrying makes progress.
echo "/workspace filesystem: $(findmnt -no FSTYPE,SOURCE -T "$WORKSPACE")"
retry() {
  local n
  for n in 1 2 3 4 5; do
    "$@" && return 0
    echo "attempt $n failed: $*" >&2; sleep 5
  done
  return 1
}

if [ "${FROZEN:-0}" = 1 ]; then
  [ -s "$BENCH_DIR/results/manifest/pip-freeze.txt" ] \
    || die "FROZEN=1 but no $BENCH_DIR/results/manifest/pip-freeze.txt: pull one from a previous pod first"
  retry uv pip install --python "$VENV/bin/python" --torch-backend=cu130 -r "$BENCH_DIR/results/manifest/pip-freeze.txt"
  retry uv pip install --python "$VENV/bin/python" --no-deps -e "${REPO}[hf]"
else
  retry uv pip install --python "$VENV/bin/python" --torch-backend=cu130 \
    "vllm==0.31.0" -e "${REPO}[hf]" -r "$BENCH_DIR/vendor/multi_turn/requirements.txt"
fi

if [ -n "${HF_TOKEN:-}" ] || [ -s "$HF_HOME/token" ]; then echo "HF token: set"; else echo "HF token: not set (anonymous rate limits)"; fi

# Resolve each model's revision once and keep it for the whole run. An existing REV_* wins.
for tag in q35-9b q3-8b; do
  ( export MODEL_TAG=$tag; source "$BENCH_DIR/env.sh"
    retry hf download "$MODEL" --revision "$MODEL_REVISION" >/dev/null
    if [ "$MODEL_REVISION" = main ]; then
      sha=$(cat "$HF_HOME/hub/models--${MODEL//\//--}/refs/main")
      var=$( [ "$tag" = q35-9b ] && echo REV_Q35_9B || echo REV_Q3_8B )
      echo "$var=\${$var:-$sha}" >> "$M1/revisions.env"
      echo "$MODEL pinned to $sha"
    fi )
done

# ShareGPT V3, the file vLLM's own benchmarks use. The hash was measured on 2026-10-16.
SHAREGPT_SHA256=35f0e213ce091ed9b9af2a1f0755e9d39f9ccec34ab281cd4ca60d70f6479ba4
# Ask the library for the path: newer `hf download` prints "  path: <file>", not the bare path.
RAW=$(retry python -c 'from huggingface_hub import hf_hub_download as d; print(d(
  "anon8231489123/ShareGPT_Vicuna_unfiltered", "ShareGPT_V3_unfiltered_cleaned_split.json",
  repo_type="dataset"))' | tail -1)
got=$(sha256sum "$RAW" | cut -d' ' -f1)
[ "$got" = "$SHAREGPT_SHA256" ] || die "ShareGPT file changed upstream: sha256 $got"
[ -s "$DATA/pg1184.txt" ] || curl -sfL -o "$DATA/pg1184.txt" https://www.gutenberg.org/ebooks/1184.txt.utf-8
sha256sum "$DATA/pg1184.txt" > "$DATA/pg1184.txt.sha256"

if [ ! -s "$DATA/sharegpt.json" ]; then
  # shellcheck disable=SC2086
  python "$BENCH_DIR/prep_sharegpt.py" "$RAW" "$DATA/sharegpt.json.tmp" $SHAREGPT_PREP_FLAGS \
    > "$LOGS/prep-sharegpt.log" 2>&1 || { tail -20 "$LOGS/prep-sharegpt.log"; die "ShareGPT conversion failed"; }
  mv "$DATA/sharegpt.json.tmp" "$DATA/sharegpt.json"
fi
n=$(python -c 'import json,sys; print(len(json.load(open(sys.argv[1]))))' "$DATA/sharegpt.json")
(( n >= 2000 )) || die "only $n ShareGPT conversations after filtering"
sha256sum "$DATA/sharegpt.json" > "$DATA/sharegpt.json.sha256"
# Same converter, flags and Python 3.12 on a laptop on 2026-10-16. A different hash means the
# sample differs from the one the planned numbers were based on; note it in the run record.
grep -q '^b5582436925c8ee8c5dd6e6789f465c5f7228450cda4176b62bc634990220134 ' "$DATA/sharegpt.json.sha256" \
  || log "WARNING: converted ShareGPT differs from the 2026-10-16 laptop conversion"

{ date -u; uname -r
  nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv,noheader
  nvcc --version | grep release
  python -c "import torch,transformers,vllm;print('vllm',vllm.__version__,'torch',torch.__version__,'transformers',transformers.__version__)" 2>&1 | tail -1
  cat "$M1/revisions.env"
  echo "repo $(cat "$REPO/GIT_SHA" 2>/dev/null || echo unknown)"
  echo "sharegpt raw $SHAREGPT_SHA256, converted $n conversations"
  cat "$DATA/sharegpt.json.sha256" "$DATA/pg1184.txt.sha256"
  echo "/workspace $(findmnt -no FSTYPE,SOURCE -T "$WORKSPACE")"
} | tee "$M1/env-manifest.txt"
uv pip freeze --python "$VENV/bin/python" > "$M1/pip-freeze.txt"
echo "bootstrap done. Next: $BENCH_DIR/bringup.sh q35-9b"
