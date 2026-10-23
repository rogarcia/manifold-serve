#!/usr/bin/env bash
# Diagnostic for bringup.sh's router overhead step: the same 200 + 200 requests, in a chosen
# order, written to $RESULTS/overhead-check/LABEL/ so the bring-up results stay as they are.
# Starts the replicas itself (a Modal call always begins with none running).
# Usage: overhead.sh LABEL [ORDER]
#   ORDER: "direct router" (bringup.sh's order, the default) or "router direct".
#   MODEL_TAG (q35-9b), OVERHEAD_N (200).
set -euo pipefail
LABEL=${1:?usage: overhead.sh LABEL [ORDER]}
ORDER=${2:-direct router}
export MODEL_TAG=${MODEL_TAG:-q35-9b}
source "$(dirname "$0")/../env.sh"
source "$BENCH_DIR/lib.sh"

"$BENCH_DIR/serve.sh" "$MODEL_TAG" nocap
load_current
OUT=$RESULTS/overhead-check/$LABEL
mkdir -p "$OUT"
{
  echo "# overhead check $LABEL, order: $ORDER, $(date -u +%FT%TZ)"
  echo "nproc $(nproc); cpu.max $(cat /sys/fs/cgroup/cpu.max 2>/dev/null || echo n/a)"
} > "$OUT/facts.txt"

# The same command as bringup.sh: one-token replies, only TTFT is compared.
N=${OVERHEAD_N:-200}
OVH=(python "$BENCH_DIR/run_bench.py"
  -i "$DATA/sharegpt.json" -m "$TOKENIZER_PATH" --served-model-name "$MODEL"
  -p 1 -k 1 -n "$N" --max-turns 2 --seed 0 --limit-min-tokens 1 --limit-max-tokens 1
  --send-conversation-id)
for way in $ORDER; do
  case "$way" in
    direct)
      wait_idle 60; reset_prefix_cache "$URL_A"
      "${OVH[@]}" -u "$URL_A" --stats-json-output "$OUT/overhead-direct.json" > "$OUT/overhead-direct.log" 2>&1 ;;
    router)
      ROUTER_BACKENDS=$URL_A "$BENCH_DIR/router.sh" prefix_aware "$OUT/overhead-router"
      wait_idle 60; reset_prefix_cache "$URL_A"
      "${OVH[@]}" -u "$ROUTER_URL" --stats-json-output "$OUT/overhead-router.json" > "$OUT/overhead-router.log" 2>&1
      scrape "$ROUTER_URL" > "$OUT/overhead-router/router_metrics.txt" || true
      "$BENCH_DIR/router.sh" stop ;;
    *) die "unknown pass: $way (direct | router)" ;;
  esac
  log "$way pass done"
done

python - "$OUT" <<'EOF' | tee -a "$OUT/facts.txt"
import json, statistics, sys
out = sys.argv[1]
p50 = {}
for way in ("direct", "router"):
    ttft = [r["ttft_ms"] for r in json.load(open(f"{out}/overhead-{way}.json")) if r["ttft_ms"] > 0]
    p50[way] = statistics.median(ttft)
    print(f"## router overhead: {way:6s} n={len(ttft)} TTFT p50 {p50[way]:.2f} ms")
print(f"## router overhead: p50 delta {p50['router'] - p50['direct']:+.2f} ms")
EOF
"$BENCH_DIR/stop.sh"
