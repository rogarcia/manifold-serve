#!/usr/bin/env bash
# Run one measurement cell against the replicas that serve.sh started.
# Usage: cell.sh RUN WORKLOAD POLICY CLIENTS SEED NREQ
#   WORKLOAD: sharegpt | w0 (ShareGPT first turns only) | w2 (shared-prefix synthetic)
#   Model tag and pressure come from $M1/current.env, so the path always matches the servers.
#   LOAD_LABEL (e.g. 70) is recorded in meta.json.
# Output: $M1/results/RUN/<model_tag>/<pressure>/WORKLOAD/POLICY-cCLIENTS-sSEED/ with DONE last.
# A cell with DONE is skipped. A partial cell is moved aside and run again from the start.
set -euo pipefail
[ $# -eq 6 ] || { sed -n '2,8p' "$0" >&2; exit 2; }
RUN=$1 WORKLOAD=$2 POLICY=$3 CLIENTS=$4 SEED=$5 NREQ=$6
source "$(dirname "$0")/env.sh"
source "$BENCH_DIR/lib.sh"

case "$POLICY" in round_robin|least_load|session_hash|prefix_aware) ;; *) die "unknown policy: $POLICY" ;; esac
case "$WORKLOAD" in
  sharegpt) INPUT=$DATA/sharegpt.json; EXTRA=() ;;
  # The bench counts messages: 2 is one user turn and its answer.
  w0)       INPUT=$DATA/sharegpt.json; EXTRA=(--max-turns 2) ;;
  w2)       INPUT=$BENCH_DIR/w2-shared-prefix.json; EXTRA=() ;;
  *) die "unknown workload: $WORKLOAD (sharegpt | w0 | w2)" ;;
esac
for n in "$CLIENTS" "$SEED" "$NREQ"; do [[ "$n" =~ ^[0-9]+$ ]] || die "CLIENTS, SEED and NREQ must be integers"; done
(( CLIENTS >= 1 && NREQ >= CLIENTS )) || die "need CLIENTS >= 1 and NREQ >= CLIENTS"
load_current

CELL=$RESULTS/$RUN/$CUR_MODEL_TAG/$CUR_PRESSURE/$WORKLOAD/$POLICY-c$CLIENTS-s$SEED
if [ -f "$CELL/DONE" ]; then log "skip (DONE): $CELL"; exit 0; fi
if [ -d "$CELL" ]; then mv "$CELL" "$CELL.partial-$(date -u +%Y%m%dT%H%M%SZ)"; fi
mkdir -p "$CELL"
[ -f "$INPUT" ] || die "missing input $INPUT"
log "cell $CELL"

# The bench gives each client floor(K / CLIENTS) conversations, so pass the multiple of CLIENTS
# nearest to ACTIVE and record both.
PER=$(( (ACTIVE + CLIENTS / 2) / CLIENTS )); (( PER >= 1 )) || PER=1
K_EFF=$(( PER * CLIENTS ))
EXPECTED=$(( CLIENTS * (NREQ / CLIENTS) ))

SAMPLER_PID=""
cleanup() { if [ -n "$SAMPLER_PID" ]; then kill "$SAMPLER_PID" 2>/dev/null || true; fi; }
trap cleanup EXIT

"$BENCH_DIR/router.sh" "$POLICY" "$CELL"

wait_idle 300
reset_prefix_cache "$URL_A"; reset_prefix_cache "$URL_B"
wait_idle 60
canary "$URL_A" || die "canary failed on replica A"
canary "$URL_B" || die "canary failed on replica B"

scrape "$URL_A" > "$CELL/metrics_before_a.txt" || die "metrics scrape failed on $URL_A"
scrape "$URL_B" > "$CELL/metrics_before_b.txt" || die "metrics scrape failed on $URL_B"
python "$BENCH_DIR/sampler.py" "$CELL/samples.tsv" "a=$URL_A" "b=$URL_B" &
SAMPLER_PID=$!

BENCH=(python "$BENCH_DIR/run_bench.py"
  -i "$INPUT" -m "$TOKENIZER_PATH" --served-model-name "$MODEL" -u "$ROUTER_URL"
  -p "$CLIENTS" -k "$K_EFF" -n "$NREQ" --send-conversation-id --seed "$SEED"
  --limit-max-tokens 0 --request-timeout-sec "$REQUEST_TIMEOUT_S"
  --stats-json-output "$CELL/stats.json" "${EXTRA[@]}")
printf '%q ' "${BENCH[@]}" > "$CELL/command.txt"; echo >> "$CELL/command.txt"
# The synthetic config names pg1184.txt relative to the working directory.
rc=0
T_START=$(date -u +%s)
(cd "$DATA" && with_timeout "${BENCH_TIMEOUT_S:-$(( DURATION_S * 4 + 900 ))}" "${BENCH[@]}") > "$CELL/bench.log" 2>&1 || rc=$?
T_END=$(date -u +%s)
# On Linux the client processes are forks with the same command line; a timeout leaves them running.
if (( rc == 124 )); then pkill -f "$BENCH_DIR/run_bench.py" || true; fi
kill "$SAMPLER_PID" 2>/dev/null || true; SAMPLER_PID=""
scrape "$URL_A" > "$CELL/metrics_after_a.txt" || die "metrics scrape failed on $URL_A"
scrape "$URL_B" > "$CELL/metrics_after_b.txt" || die "metrics scrape failed on $URL_B"
scrape "$ROUTER_URL" > "$CELL/router_metrics_after.txt" || true
"$BENCH_DIR/router.sh" stop
(( rc == 0 )) || { tail -30 "$CELL/bench.log" >&2; die "bench exited $rc (124 = timeout)"; }

GIT_SHA=$(cat "$REPO/GIT_SHA" 2>/dev/null || git -C "$REPO" describe --always --dirty 2>/dev/null || echo unknown)
export RUN WORKLOAD POLICY CLIENTS SEED NREQ K_EFF EXPECTED T_START T_END INPUT CELL GIT_SHA MODEL MODEL_REVISION LOAD_LABEL="${LOAD_LABEL:-}"
python - <<'EOF' || die "meta.json not written, or the bench did not end cleanly (see above)"
import hashlib, json, os, sys
e = os.environ
cell = e["CELL"]
records = json.load(open(os.path.join(cell, "stats.json")))
decisions = sum(1 for _ in open(os.path.join(cell, "decisions.jsonl")))
current = dict(l.strip().split("=", 1) for l in open(os.path.join(e["M1"], "current.env")) if "=" in l)
meta = {
    "run": e["RUN"], "model_tag": current["CUR_MODEL_TAG"], "pressure": current["CUR_PRESSURE"],
    "workload": e["WORKLOAD"], "policy": e["POLICY"], "clients": int(e["CLIENTS"]),
    "seed": int(e["SEED"]), "nreq": int(e["NREQ"]), "load_label": e["LOAD_LABEL"] or None,
    "active_requested": int(e["ACTIVE"]), "active_effective": int(e["K_EFF"]),
    "expected_requests": int(e["EXPECTED"]), "bench_records": len(records),
    "decision_lines": decisions,
    "command": open(os.path.join(cell, "command.txt")).read().strip(),
    "git_sha": e["GIT_SHA"], "model": e["MODEL"], "model_revision": e["MODEL_REVISION"],
    "input": e["INPUT"], "input_sha256": hashlib.sha256(open(e["INPUT"], "rb").read()).hexdigest(),
    "t_start": int(e["T_START"]), "t_end": int(e["T_END"]),
    "env": {k: e.get(k) for k in ("MAX_MODEL_LEN", "MAX_NUM_BATCHED_TOKENS", "GPU_MEMORY_UTILIZATION",
                                  "WARMUP_S", "DURATION_S", "REQUEST_TIMEOUT_S", "SIM")} | current,
}
json.dump(meta, open(os.path.join(cell, "meta.json"), "w"), indent=2)
# The bench stops every client once the first reaches its quota, so a clean run returns somewhat
# fewer records than requested. A crashed client, or one that ran out of conversations, is not clean.
log = open(os.path.join(cell, "bench.log"), errors="replace").read()
problems = [p for p, bad in (
    ("no client reached its request limit", "reached request limit" not in log),
    ("a client process exited non-zero", "exited with exit code" in log),
    (f"under MIN_COMPLETE={e['MIN_COMPLETE']} of the expected requests",
     len(records) < float(e["MIN_COMPLETE"]) * int(e["EXPECTED"])),
) if bad]
print(f"{len(records)} bench records of {e['EXPECTED']} expected, {decisions} decision lines"
      + "".join(f"; {p}" for p in problems), file=sys.stderr)
sys.exit(3 if problems else 0)
EOF
touch "$CELL/DONE"
log "done: $CELL"
