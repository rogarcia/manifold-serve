# Sourced by every m0-redux script. Non-login SSH shells on the RunPod image
# lack the CUDA bin dir, and vLLM needs the venv's cu13 libs on LD_LIBRARY_PATH.
export WORKSPACE=/workspace
export VENV=$WORKSPACE/.venv
export M0=$WORKSPACE/m0-redux
export HF_HOME=$WORKSPACE/hf
export PATH="$VENV/bin:$HOME/.local/bin:/usr/local/cuda/bin:$PATH"
export LD_LIBRARY_PATH="$VENV/lib/python3.12/site-packages/nvidia/cu13/lib:${LD_LIBRARY_PATH:-}"
export VLLM_USE_RUST_FRONTEND=1
# uv's cache is on container disk and the venv on /workspace, so hardlinks always fail; copy directly.
export UV_LINK_MODE=copy

# Optional secrets (HF_TOKEN=...). Not versioned; see .env.example.
# `hf auth login` with HF_HOME set works too: it stores the token at $HF_HOME/token.
if [ -f "$M0/.env" ]; then set -a; . "$M0/.env"; set +a; fi

export MODEL=nvidia/Qwen3.8-27B-NVFP4
# Revision used for the 2026-09-21 results.
export MODEL_REVISION=${MODEL_REVISION:-482ca0f3832238542f8f5295dde86b5f22711d80}
export PORT=8000
# Recipe value. Lower it (e.g. MAX_MODEL_LEN=32768) if KV budget becomes a confound.
export MAX_MODEL_LEN=${MAX_MODEL_LEN:-262144}
