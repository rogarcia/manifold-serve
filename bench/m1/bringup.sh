#!/usr/bin/env bash
# Bring-up for one model: serve with vLLM sizing the KV pool, record the startup facts, check
# the prefix cache and that thinking is off, and measure router overhead.
# Usage: bringup.sh MODEL_TAG
#   Writes $M1/bringup-<tag>.env (read by env.sh) and $M1/results/bringup/<tag>/.
#   SKIP_SERVE=1 uses the replicas already running. SKIP_OVERHEAD=1 skips the router overhead
#   measurement. OVERHEAD_N (default 200) requests each way.
set -euo pipefail
export MODEL_TAG=${1:?usage: bringup.sh MODEL_TAG}
source "$(dirname "$0")/env.sh"
source "$BENCH_DIR/lib.sh"

[ "${SKIP_SERVE:-0}" = 1 ] || "$BENCH_DIR/serve.sh" "$MODEL_TAG" nocap
load_current
[ "$CUR_MODEL_TAG" = "$MODEL_TAG" ] || die "replicas serve $CUR_MODEL_TAG, not $MODEL_TAG"
[ "$CUR_PRESSURE" = nocap ] || die "bring-up needs the uncapped pool (serving $CUR_PRESSURE)"
OUT=$RESULTS/bringup/$MODEL_TAG
mkdir -p "$OUT"
cp "$CUR_SERVE_LOG_A" "$OUT/serve-a.log"; cp "$CUR_SERVE_LOG_B" "$OUT/serve-b.log"

eval "$(parse_serve_log "$CUR_SERVE_LOG_A")"
KV_TOKENS=$(( CUR_KV_TOKENS_A < CUR_KV_TOKENS_B ? CUR_KV_TOKENS_A : CUR_KV_TOKENS_B ))
BYTES_PER_TOKEN=""
[ -n "$KV_GIB" ] && BYTES_PER_TOKEN=$(awk -v g="$KV_GIB" -v t="$KV_TOKENS" 'BEGIN { printf "%.1f", g * 1073741824 / t }')
cat > "$M1/bringup-$MODEL_TAG.env" <<EOF
BLOCK_TOKENS=$BLOCK_TOKENS
BLOCK_SOURCE=$BLOCK_SOURCE
KV_TOKENS=$KV_TOKENS
KV_GIB=$KV_GIB
BYTES_PER_TOKEN=$BYTES_PER_TOKEN
CUDAGRAPH_SIZES='$CUDAGRAPH_SIZES'
EOF
cp "$M1/bringup-$MODEL_TAG.env" "$OUT/"

FACTS=$OUT/bringup-facts.txt
{
  echo "# bring-up $MODEL_TAG ($MODEL @ $MODEL_REVISION), $(date -u +%FT%TZ)"
  echo "## parsed"; cat "$M1/bringup-$MODEL_TAG.env"
  for r in a b; do
    echo "## replica $r startup lines"
    grep -E 'Setting attention block size|GPU KV cache size|Available KV cache memory|Maximum concurrency|block_size|num_gpu_blocks' "$OUT/serve-$r.log" | cut -c1-400 || true
  done
  if command -v nvidia-smi >/dev/null; then nvidia-smi --query-gpu=index,name,driver_version,memory.used,memory.total --format=csv; fi
} > "$FACTS"
[ "$BLOCK_SOURCE" = log ] || log "no 'Setting attention block size' line: assuming vLLM's default of 16 tokens. Confirm in $FACTS."

wait_idle 60
reset_prefix_cache "$URL_A"; reset_prefix_cache "$URL_B"
canary "$URL_A" || die "canary failed on replica A"
canary "$URL_B" || die "canary failed on replica B"
echo "## canary: prefix_cache_hits_total rose on the second pass on both replicas" >> "$FACTS"

# The router renders prompts with this tokenizer's chat template; a checkpoint without one would
# make every prefix hash wrong.
if [ "$CUR_SIM" != 1 ]; then
  python -c 'import sys; from transformers import AutoTokenizer; t = AutoTokenizer.from_pretrained(sys.argv[1]); sys.exit(0 if t.chat_template else 1)' "$TOKENIZER_PATH" \
    || die "AutoTokenizer on $TOKENIZER_PATH has no chat template"
  echo "## router tokenizer: $TOKENIZER_PATH has a chat template" >> "$FACTS"
fi

REPLY=$(chat_once "$URL_A" "What is 17 times 23? Answer in one sentence." 200 \
  | python -c 'import json,sys; print(json.load(sys.stdin)["choices"][0]["message"]["content"])')
{ echo "## thinking check reply"; echo "$REPLY"; } >> "$FACTS"
if grep -q '<think>' <<< "$REPLY"; then die "reply contains <think>: thinking is not off"; fi
log "thinking check ok: reply has no <think>"

if [ "${SKIP_OVERHEAD:-0}" != 1 ]; then
  N=${OVERHEAD_N:-200}
  OVH=(python "$BENCH_DIR/run_bench.py"
    -i "$DATA/sharegpt.json" -m "$TOKENIZER_PATH" --served-model-name "$MODEL"
    -p 1 -k 1 -n "$N" --max-turns 2 --seed 0 --limit-max-tokens 0 --send-conversation-id)
  reset_prefix_cache "$URL_A"
  "${OVH[@]}" -u "$URL_A" --stats-json-output "$OUT/overhead-direct.json" > "$OUT/overhead-direct.log" 2>&1
  ROUTER_BACKENDS=$URL_A "$BENCH_DIR/router.sh" prefix_aware "$OUT/overhead-router"
  wait_idle 60; reset_prefix_cache "$URL_A"
  "${OVH[@]}" -u "$ROUTER_URL" --stats-json-output "$OUT/overhead-router.json" > "$OUT/overhead-router.log" 2>&1
  scrape "$ROUTER_URL" > "$OUT/overhead-router/router_metrics.txt" || true
  "$BENCH_DIR/router.sh" stop
  python - "$OUT" <<'EOF' | tee -a "$FACTS"
import json, statistics, sys
out = sys.argv[1]
p50 = {}
for way in ("direct", "router"):
    ttft = [r["ttft_ms"] for r in json.load(open(f"{out}/overhead-{way}.json")) if r["ttft_ms"] > 0]
    p50[way] = statistics.median(ttft)
    print(f"## router overhead: {way:6s} n={len(ttft)} TTFT p50 {p50[way]:.2f} ms")
print(f"## router overhead: p50 delta {p50['router'] - p50['direct']:+.2f} ms")
EOF
fi
log "bring-up facts: $FACTS"
