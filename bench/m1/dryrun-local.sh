#!/usr/bin/env bash
# Laptop dry run, no GPU: the real bring-up, calibration, sizing, matrix and summary scripts
# against two `python -m manifold.sim` replicas, with small request counts.
# Usage: dryrun-local.sh [M1_DIR]     default $TMPDIR/m1-dryrun; delete it to start over.
# Needs uv, tmux and network access to Hugging Face for the Qwen3.5-9B tokenizer files.
set -euo pipefail
D=$(cd "$(dirname "$0")" && pwd)
export M1=${1:-${TMPDIR:-/tmp}/m1-dryrun}
export WORKSPACE=$M1 VENV=$M1/.venv HF_HOME=$M1/hf SIM=1 MODEL_TAG=q35-9b
# A private tmux server, so the dry run's "router" and "vllm-a" sessions never touch yours.
export TMUX_TMPDIR=$M1/tmux
export ACTIVE=${ACTIVE:-8} WARMUP_S=${WARMUP_S:-2} DURATION_S=${DURATION_S:-20} SLO_MS=${SLO_MS:-500}
export MAX_MODEL_LEN=4096 OVERHEAD_N=${OVERHEAD_N:-20}
# The vendored bench imports one logging helper from vllm, which the laptop does not have.
export PYTHONPATH=$D/tests/vllm_shim${PYTHONPATH:+:$PYTHONPATH}
export CALIB_CLIENTS=${CALIB_CLIENTS:-1 2 4} CALIB_REQ_PER_CLIENT=10 CALIB_MIN_NREQ=20
# The sim's prefill cost makes cache hits visible within a few dozen requests.
export SIM_FLAGS=${SIM_FLAGS:---max-num-seqs 8 --prefill-us-per-token 200 --decode-ms-per-token 2}
mkdir -p "$M1" "$TMUX_TMPDIR"
source "$D/env.sh"
source "$D/lib.sh"
mkdir -p "$DATA" "$LOGS"
trap '"$D/stop.sh" || true' EXIT

[ -x "$VENV/bin/python" ] || uv venv "$VENV" --python 3.12
uv pip install --python "$VENV/bin/python" -q -e "$REPO" -r "$D/vendor/multi_turn/requirements.txt"
python -c 'import manifold.sim, manifold.router' || die "manifold.sim or manifold.router is missing from $REPO"

# Tokenizer files only, pinned like bootstrap.sh pins the full model.
if [ ! -f "$M1/revisions.env" ]; then
  hf download "$MODEL" --include 'tokenizer*' --include 'vocab*' --include 'merges*' --include '*config.json' >/dev/null
  echo "REV_Q35_9B=\${REV_Q35_9B:-$(cat "$HF_HOME/hub/models--${MODEL//\//--}/refs/main")}" > "$M1/revisions.env"
  source "$D/env.sh"
fi

# A small ShareGPT-shaped file, run through the same converter flags as the pod. Some
# conversations end on a user turn and one has non-ASCII text, so both corrections are exercised.
if [ ! -s "$DATA/sharegpt.json" ]; then
  python - "$DATA/sharegpt-raw.json" <<'EOF'
import json, random, sys
rng = random.Random(0)
words = "cache prefix router replica block token latency queue prompt answer model batch".split()
def text(n): return " ".join(rng.choice(words) for _ in range(n)).capitalize() + "."
items = []
for i in range(400):
    turns = [{"from": "human" if t % 2 == 0 else "gpt", "value": text(rng.randint(40, 300) if t % 2 == 0 else rng.randint(20, 120))}
             for t in range(rng.choice([4, 5, 6, 7, 8, 10]))]
    if i == 7:
        turns[0]["value"] += " café"
    items.append({"id": f"dry{i:04d}_0", "conversations": turns})
json.dump(items, open(sys.argv[1], "w"))
EOF
  # shellcheck disable=SC2086
  python "$D/prep_sharegpt.py" "$DATA/sharegpt-raw.json" "$DATA/sharegpt.json" $SHAREGPT_PREP_FLAGS > "$LOGS/prep-sharegpt.log" 2>&1
  tail -2 "$LOGS/prep-sharegpt.log"
fi

log "bring-up"
"$D/bringup.sh" q35-9b

log "calibration"
"$D/calibrate.sh" dry-calib
if [ ! -f "$M1/knee.env" ]; then
  log "the sims never crossed the SLO; using knee 4 for the dry run"
  uv run "$D/summarize.py" --calibrate "$RESULTS/dry-calib/q35-9b/nocap" --warmup-s "$WARMUP_S" \
    --slo-ms "$SLO_MS" --duration-s "$DURATION_S" --knee 4 --write-knee "$M1/knee.env"
fi
uv run "$D/size_kv.py" "$RESULTS/dry-calib/q35-9b/nocap" --model-tag q35-9b --m1 "$M1"

log "mini matrix: both pressures, load 70%, seed 1, four policies"
PRESSURES="fits pressure" LOADS=70 SEEDS=1 "$D/matrix.sh" dry
log "rerun: every cell must be skipped"
PRESSURES="fits pressure" LOADS=70 SEEDS=1 "$D/matrix.sh" dry 2>&1 | tee "$LOGS/matrix-rerun.log"
grep -q "all cells done" "$LOGS/matrix-rerun.log" || die "rerun did not skip the finished cells"

uv run "$D/summarize.py" "$RESULTS/dry" --warmup-s "$WARMUP_S"
log "dry run complete: $M1"
