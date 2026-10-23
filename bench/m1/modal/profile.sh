#!/usr/bin/env bash
# Diagnostic, not a measurement: one cell at CLIENTS clients while sampling GPU use (nvidia-smi
# dmon) and CPU per process (top). Answers what limits throughput and how close the router and
# the vLLM front ends are to a full core. The cell's numbers are not reported. Modal does not
# allow ptrace, so py-spy cannot attach here.
# Usage: profile.sh [CLIENTS] [POLICY]     defaults 192, least_load. REQ_PER_CLIENT (40).
# Output: $RESULTS/diag/profile-POLICY-cCLIENTS/ (summary.txt, dmon.txt, top.txt) and the cell
# under $RESULTS/diag/.
set -euo pipefail
CLIENTS=${1:-192}
POLICY=${2:-least_load}
export MODEL_TAG=${MODEL_TAG:-q35-9b}
source "$(dirname "$0")/../env.sh"
source "$BENCH_DIR/lib.sh"

OUT=$RESULTS/diag/profile-$POLICY-c$CLIENTS
mkdir -p "$OUT"
"$BENCH_DIR/serve.sh" "$MODEL_TAG" nocap
{ echo "# profile $POLICY c$CLIENTS, $(date -u +%FT%TZ)"; echo "nproc $(nproc)"; uptime; } > "$OUT/summary.txt"

nvidia-smi dmon -s puc -d 1 -o T > "$OUT/dmon.txt" 2>&1 & DMON=$!
top -b -c -w 400 -d 2 > "$OUT/top.txt" 2>&1 & TOP=$!


rc=0
LOAD_LABEL=diag MIN_COMPLETE=0.6 "$BENCH_DIR/cell.sh" diag sharegpt "$POLICY" "$CLIENTS" 0 \
  $(( CLIENTS * ${REQ_PER_CLIENT:-40} )) || rc=$?
kill "$DMON" "$TOP" 2>/dev/null || true

# Mean CPU per process group over the snapshots taken while the bench ran (top's %CPU is per
# core, so 100 is one full core), and mean GPU use.
python - "$OUT" >> "$OUT/summary.txt" <<'EOF'
import re, statistics, sys
out = sys.argv[1]
groups = {"engine core": r"EngineCore", "vllm API server": r"bin/vllm serve",
          "router": r"manifold\.router", "bench clients": r"run_bench\.py", "other": r"."}
snaps, cur = [], None
for line in open(f"{out}/top.txt", errors="replace"):
    if line.startswith("top -"):
        cur = {g: 0.0 for g in groups}; snaps.append(cur); continue
    f = line.split(None, 11)
    if cur is None or len(f) < 12 or not f[0].isdigit():
        continue
    for g, pat in groups.items():
        if re.search(pat, f[11]):
            cur[g] += float(f[8]); break
busy = [s for s in snaps if s["bench clients"] > 0]
print(f"## CPU, mean %CPU per group over {len(busy)} snapshots with the bench running (100 = one core)")
for g in groups:
    vals = [s[g] for s in busy] or [0]
    print(f"{g:16s} mean {statistics.mean(vals):7.1f}  max {max(vals):7.1f}")
lines = open(f"{out}/dmon.txt").read().splitlines()
head = next((l.lstrip("#").split() for l in lines if l.startswith("#") and "gpu" in l.split()), None)
rows = [l.split() for l in lines if l.strip() and not l.startswith("#")]
print("## GPU (nvidia-smi dmon, whole cell, means)")
if head and rows:
    ix = {name: i for i, name in enumerate(head)}
    def mean(rs, name):
        v = [float(r[ix[name]]) for r in rs if name in ix and r[ix[name]].replace(".", "", 1).isdigit()]
        return f"{statistics.mean(v):.1f}" if v else "n/a"
    for gpu in sorted({r[ix["gpu"]] for r in rows}):
        rs = [r for r in rows if r[ix["gpu"]] == gpu]
        print(f"gpu {gpu}: " + "  ".join(f"{n} {mean(rs, n)}" for n in ("pwr", "sm", "mem", "pclk", "mclk")))
else:
    print("no dmon samples; see dmon.txt")
EOF
"$BENCH_DIR/stop.sh"
cat "$OUT/summary.txt"
exit "$rc"
