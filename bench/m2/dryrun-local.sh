#!/usr/bin/env bash
# Laptop dry run, no GPU and no Hugging Face: the probe, every cell and the summary against
# manifold.sim, with the trace compressed 20 times, a kill -9 of the autoscaler in C1 and a reclaim
# in both C1 and C5. It proves the pipeline runs; its numbers say nothing about vLLM.
# Usage: dryrun-local.sh [DIR]     default $TMPDIR/m2-dryrun; delete it to start over. About 5 minutes.
set -euo pipefail
D=$(cd "$(dirname "$0")" && pwd)
export WORKSPACE=${1:-${TMPDIR:-/tmp}/m2-dryrun}
REPO=$(cd "$D/../.." && pwd)
export REPO M1=$WORKSPACE/m1 M2=$WORKSPACE/m2 M2_RUN_DIR=$WORKSPACE/run
export VENV=${VENV:-$REPO/.venv} HF_HOME=$WORKSPACE/hf PYTHONPATH=$REPO SIM=1 MODEL_TAG=q35-9b
# stop.sh looks for tmux sessions by name; a private server keeps it away from yours.
export TMUX_TMPDIR=$WORKSPACE/tmux
export TIME_SCALE=20 RUN=dry LOADGEN_WORKERS=2 SIM_START_S=1 RECLAIM_GRACE_S=3
export HOLD_INTERACTIVE_S=30 HOLD_BATCH_S=60
export SIM_FLAGS="--prefill-us-per-token 12 --kv-blocks 65536"
export SPEC_OVERRIDE='{"tick_s": 0.25, "target_interactive": 26, "up_window_s": 0.5, "settle_s": 2, "down_window_s": 3, "drain_deadline_s": 3}'
mkdir -p "$M1/data" "$M2" "$TMUX_TMPDIR"
source "$D/common.sh"
trap stop_all EXIT

python - "$DATA/sharegpt.json" <<'EOF'
import json, random, sys
rng = random.Random(0)
words = "cache prefix router replica block token latency queue prompt answer model batch".split()
def text(n): return " ".join(rng.choice(words) for _ in range(n)).capitalize() + "."
items = []
for i in range(1200):
    n_turns = rng.choice([1, 2, 3, 4, 5, 8])
    messages = []
    for _ in range(n_turns):
        messages.append({"role": "user", "content": text(rng.randint(40, 300))})
        messages.append({"role": "assistant", "content": text(rng.randint(100, 300))})
    items.append({"id": f"dry{i:04d}", "messages": messages})
json.dump(items, open(sys.argv[1], "w"))
EOF

log "probe"
"$D/probe.sh"
cat "$M2/probe.env" >&2

log "C1 with kill -9 of the autoscaler at t=400 and a reclaim of slot 1 at t=600"
EVENTS="400 kill_autoscaler;600 reclaim 1 3" GATE=on "$D/cell.sh" c1-reconcile-s0 reconcile 0

log "the rest of the plan"
BUDGET_MIN=600 "$D/plan.sh" dry
log "rerun: every step must be skipped"
BUDGET_MIN=600 "$D/plan.sh" dry 2>&1 | tee "$M2/rerun.log"
grep -q "0 ran, 7 skipped as done" "$M2/rerun.log" || die "rerun did not skip the finished cells"

python - "$M2/results/dry" <<'EOF'
import json, sys
from pathlib import Path
run = Path(sys.argv[1])
ticks = [json.loads(x) for x in (run / "c1-reconcile-s0/ticks.jsonl").read_text().splitlines()]
ready = [any(p == "ready" for p in t["phases"]) for t in ticks]
rises = sum(b and not a for a, b in zip(ready, ready[1:]))
assert rises >= 2, f"expected a first and a second cold start, saw {rises}"
assert any(t["run"] == 2 for t in ticks), "the second replica was never requested"
events = (run / "c1-reconcile-s0/events.jsonl").read_text().splitlines()
assert len(events) == 2, events
print(f"cold starts {rises}, events {len(events)}")
EOF
grep -q "NOT EVALUATED" "$M2/results/dry/summary.txt" && die "a rule was not evaluated"
log "dry run complete: $M2/results/dry/summary.txt"
