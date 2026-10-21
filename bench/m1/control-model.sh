#!/usr/bin/env bash
# Dense control: the headline cells (70% and 95% of the knee, under pressure, four policies) on
# Qwen3-8B, which has full attention on every layer and vLLM's default block size.
# Client and request counts are the primary model's (knee.env). The pressure pool is sized from
# the primary model's calibration working set with this model's block size and bytes per token.
# Usage: control-model.sh RUN [CALIB_ROOT]     CALIB_ROOT defaults to $M1/results/calib/q35-9b/nocap
#   SEEDS (default 1).
set -euo pipefail
RUN=${1:?usage: control-model.sh RUN [CALIB_ROOT]}
export MODEL_TAG=q3-8b
source "$(dirname "$0")/env.sh"
source "$BENCH_DIR/lib.sh"
CALIB_ROOT=${2:-$RESULTS/calib/q35-9b/nocap}

[ -f "$M1/bringup-$MODEL_TAG.env" ] || SKIP_OVERHEAD=1 "$BENCH_DIR/bringup.sh" "$MODEL_TAG"
[ -f "$M1/pressure-$MODEL_TAG.env" ] || uv run "$BENCH_DIR/size_kv.py" "$CALIB_ROOT" --model-tag "$MODEL_TAG" --m1 "$M1"
PRESSURES=pressure LOADS="70 95" SEEDS=${SEEDS:-1} "$BENCH_DIR/matrix.sh" "$RUN"
