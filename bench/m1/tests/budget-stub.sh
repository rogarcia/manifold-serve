#!/usr/bin/env bash
# Exercises budget.sh plan order, restarts, skip-if-DONE, deadline skips and the failure stop with
# no servers: a fake bench dir whose serve.sh, cell.sh, stop.sh and curl are stubs.
# Usage: tests/budget-stub.sh
# shellcheck disable=SC2016,SC2034,SC2001  # check() evals its single-quoted assertions
set -uo pipefail
SRC=$(cd "$(dirname "$0")/.." && pwd)
T=$(mktemp -d "${TMPDIR:-/tmp}/m1-budget-stub.XXXXXX")
trap 'rm -rf "$T"' EXIT
B=$T/repo/bench/m1
mkdir -p "$B" "$T/venv/bin"
cp "$SRC/env.sh" "$SRC/lib.sh" "$SRC/budget.sh" "$B/"
export M1=$T/m1 WORKSPACE=$T VENV=$T/venv HF_HOME=$T/hf
mkdir -p "$M1/logs"
echo 'CLIENTS_95=182 NREQ_95=8521 CLIENTS_70=134 NREQ_70=7690' > "$M1/knee.env"
# Replicas answer /health only between a serve and a stop.
printf '#!/usr/bin/env bash\n[ -f "$M1/up" ]\n' > "$VENV/bin/curl"
cat > "$B/serve.sh" <<'EOF'
#!/usr/bin/env bash
echo "serve $2" >> "$M1/calls"; sleep 1
printf 'CUR_MODEL_TAG=%s\nCUR_PRESSURE=%s\n' "$1" "$2" > "$M1/current.env"; touch "$M1/up"
EOF
cat > "$B/cell.sh" <<'EOF'
#!/usr/bin/env bash
echo "cell $*" >> "$M1/calls"; sleep 1
[ -z "${CELL_FAIL:-}" ] || exit 1
. "$M1/current.env"; d=$M1/results/$1/$CUR_MODEL_TAG/$CUR_PRESSURE/$2/$3-c$4-s$5
mkdir -p "$d" && touch "$d/DONE"
EOF
printf '#!/usr/bin/env bash\necho stop >> "$M1/calls"; rm -f "$M1/up"\n' > "$B/stop.sh"
chmod +x "$VENV/bin/curl" "$B"/*.sh
fail=0
check() { if eval "$2"; then echo "ok    $1"; else echo "FAIL  $1"; fail=1; fi; }
run() { rm -f "$M1/calls"; out=$("$B/budget.sh" "$@" 2>&1); rc=$?; echo "$out" | sed 's/^/    /'; }

echo "(a) large budget, 1 s estimates"
BUDGET_MIN=100 CELL_EST_S=1 RESTART_EST_S=1 run r1
check "exit 0" '[ $rc -eq 0 ]'
check "3 serves: pressure, fits, pressure" '[ "$(grep serve "$M1/calls" | tr "\n" " ")" = "serve pressure serve fits serve pressure " ]'
check "12 cells in plan order" '[ "$(grep cell "$M1/calls" | awk "{print \$4, \$6, \$5}" | tr "\n" ",")" = "least_load 1 182,prefix_aware 1 182,session_hash 1 182,round_robin 1 182,round_robin 2 182,session_hash 2 182,prefix_aware 2 182,least_load 2 182,least_load 1 182,prefix_aware 1 182,least_load 1 134,prefix_aware 1 134," ]'
check "stop.sh ran last" '[ "$(tail -1 "$M1/calls")" = stop ]'

echo "(b) rerun"
BUDGET_MIN=100 run r1
check "exit 0, all 12 skipped as done" '[ $rc -eq 0 ] && [ "$(grep -c "skip (DONE)" <<< "$out")" = 12 ]'
check "no serve, no cell" '! grep -qE "serve|cell" "$M1/calls"'

echo "(c) 60 s budget, 40 s estimates, fits already serving"
printf 'pressure 95 least_load 1  # needs a restart: 80 s\n\nfits 95 least_load 1\npressure 95 round_robin 1\n' > "$T/plan"
printf 'CUR_MODEL_TAG=q35-9b\nCUR_PRESSURE=fits\n' > "$M1/current.env"; touch "$M1/up"
PLAN_FILE=$T/plan BUDGET_MIN=1 CELL_EST_S=40 RESTART_EST_S=40 run r2
check "line 1 skipped for budget" 'grep -q "skip (budget): pressure 95 least_load 1, needs 80s" <<< "$out"'
check "line 2 ran on fits with no restart" '[ "$(sed -n 1p "$M1/calls")" = "cell r2 sharegpt least_load 182 1 8521" ]'
check "line 3 fits once the cell estimate is the 1 s seen" '[ "$(sed -n 2,3p "$M1/calls" | tr "\n" ,)" = "serve pressure,cell r2 sharegpt round_robin 182 1 8521," ]'

echo "(d) every cell fails"
CELL_FAIL=1 BUDGET_MIN=100 CELL_EST_S=1 RESTART_EST_S=1 run r3
check "non-zero exit after 2 failed cells" '[ $rc -ne 0 ] && [ "$(grep -c cell "$M1/calls")" = 2 ]'
check "both in failed-cells.txt" '[ "$(grep -c " r3 " "$M1/logs/failed-cells.txt")" = 2 ]'
check "stop.sh still ran" '[ "$(tail -1 "$M1/calls")" = stop ]'

echo "(e) no BUDGET_MIN"
"$B/budget.sh" r1 >/dev/null 2>&1; rc=$?
check "exit 2" '[ $rc -eq 2 ]'

[ "$fail" = 0 ] && echo "budget-stub: all checks passed"
exit "$fail"
