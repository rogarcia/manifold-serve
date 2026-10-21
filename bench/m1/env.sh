# shellcheck shell=bash disable=SC2034  # the scripts that source this file use these variables
# Sourced by every bench/m1 script. Set MODEL_TAG (and any override below) before sourcing.
# Non-login SSH shells on the RunPod image lack the CUDA bin dir, and vLLM needs the venv's cu13
# libs on LD_LIBRARY_PATH.
BENCH_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export BENCH_DIR
export REPO=${REPO:-$(cd "$BENCH_DIR/../.." && pwd)}
export WORKSPACE=${WORKSPACE:-/workspace}
export M1=${M1:-$WORKSPACE/m1}
export VENV=${VENV:-$WORKSPACE/.venv}
export HF_HOME=${HF_HOME:-$WORKSPACE/hf}
export DATA=$M1/data RESULTS=$M1/results LOGS=$M1/logs
export PATH="$VENV/bin:$HOME/.local/bin:/usr/local/cuda/bin:$PATH"
export LD_LIBRARY_PATH="$VENV/lib/python3.12/site-packages/nvidia/cu13/lib:${LD_LIBRARY_PATH:-}"
# uv's cache is on container disk and the venv on /workspace, so hardlinks always fail; copy directly.
export UV_LINK_MODE=copy
# Enables POST /reset_prefix_cache on vLLM.
export VLLM_SERVER_DEV_MODE=1

# Optional secrets (HF_TOKEN=...). Not versioned; see .env.example.
if [ -f "$M1/.env" ]; then set -a; . "$M1/.env"; set +a; fi
# Commit SHAs resolved by bootstrap.sh, so every serve in a run uses the same weights. The file
# only fills REV_Q35_9B / REV_Q3_8B when unset, so `REV_Q3_8B=<sha> ./serve.sh q3-8b` overrides.
if [ -f "$M1/revisions.env" ]; then . "$M1/revisions.env"; fi

export MODEL_TAG=${MODEL_TAG:-q35-9b}
case "$MODEL_TAG" in
  # Multimodal checkpoint: serve the text model only.
  q35-9b) MODEL=Qwen/Qwen3.5-9B; MODEL_FLAGS="--language-model-only"; REV_VAR=REV_Q35_9B ;;
  q3-8b)  MODEL=Qwen/Qwen3-8B;   MODEL_FLAGS="";                      REV_VAR=REV_Q3_8B ;;
  *) echo "unknown MODEL_TAG: $MODEL_TAG (q35-9b | q3-8b)" >&2; return 2 2>/dev/null || exit 2 ;;
esac
# Derived from MODEL_TAG on every source and not exported, so a child run for another model
# never inherits them.
MODEL_REVISION=${!REV_VAR:-main}
# Pinned local snapshot when bootstrap has downloaded it; the repo id otherwise (dry runs).
TOKENIZER_PATH=$HF_HOME/hub/models--${MODEL//\//--}/snapshots/$MODEL_REVISION
[ -d "$TOKENIZER_PATH" ] || TOKENIZER_PATH=$MODEL
# Thinking is on by default for both models. Off for every request, at the server and the router.
export CHAT_TEMPLATE_KWARGS='{"enable_thinking": false}'

export PORT_A=${PORT_A:-8001} PORT_B=${PORT_B:-8002} ROUTER_PORT=${ROUTER_PORT:-8000}
export URL_A=http://localhost:$PORT_A URL_B=http://localhost:$PORT_B ROUTER_URL=http://localhost:$ROUTER_PORT
export MAX_MODEL_LEN=${MAX_MODEL_LEN:-32768}
export MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS:-8192}
export GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.90}

# Live conversations in the bench. Fixed across load levels so the working set stays fixed.
export ACTIVE=${ACTIVE:-64}
export WARMUP_S=${WARMUP_S:-60}
export SLO_MS=${SLO_MS:-1000}
export DURATION_S=${DURATION_S:-420}
export REQUEST_TIMEOUT_S=${REQUEST_TIMEOUT_S:-300}
# A cell fails if the bench returns fewer than this share of the requests it was asked for.
export MIN_COMPLETE=${MIN_COMPLETE:-0.8}

# ShareGPT conversion (prep_sharegpt.py): multi-turn only (at least two user turns), at most 10
# user turns, every message 1 to 4,000 characters. That bounds a conversation at about
# 20 x 1,000 tokens, under MAX_MODEL_LEN 32,768, and the replayed answers are capped at the
# dataset answer length so they cannot grow it.
SHAREGPT_PREP_FLAGS="--seed 0 --min-turns 4 --max-turns 20 --min-content-len 1 --max-content-len 4000 --max-items 4000"

# SIM=1 runs `python -m manifold.sim` in place of vLLM (laptop dry run).
export SIM=${SIM:-0}
export SIM_BLOCK_TOKENS=${SIM_BLOCK_TOKENS:-16}
export SIM_KV_BLOCKS=${SIM_KV_BLOCKS:-4096}
export SIM_FLAGS=${SIM_FLAGS:-}
# The sim counts UTF-8 bytes; size_kv.py counts model tokens. English text is about 4 bytes a token.
export SIM_BYTES_PER_TOKEN=${SIM_BYTES_PER_TOKEN:-4}

# Written by bringup.sh, size_kv.py and summarize.py calibrate. Per model where the value is.
for f in "$M1/bringup-$MODEL_TAG.env" "$M1/pressure-$MODEL_TAG.env" "$M1/knee.env"; do
  # shellcheck source=/dev/null
  if [ -f "$f" ]; then . "$f"; fi
done

# Names re-exported into processes started in tmux, whose environment is the tmux server's.
LAUNCH_VARS="REPO WORKSPACE M1 VENV HF_HOME MODEL_TAG REV_Q35_9B REV_Q3_8B PORT_A PORT_B ROUTER_PORT MAX_MODEL_LEN MAX_NUM_BATCHED_TOKENS GPU_MEMORY_UTILIZATION SIM SIM_BLOCK_TOKENS SIM_KV_BLOCKS SIM_FLAGS SIM_BYTES_PER_TOKEN"
