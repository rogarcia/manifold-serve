#!/usr/bin/env bash
# Exercises cell.sh argument checks, skip-if-DONE and partial-cell handling with no servers:
# a copy of bench/m1 whose router.sh is a stub that fails.
# Usage: tests/cell-stub.sh
# shellcheck disable=SC2016,SC2034  # check() evals its single-quoted assertions
set -uo pipefail
SRC=$(cd "$(dirname "$0")/.." && pwd)
T=$(mktemp -d "${TMPDIR:-/tmp}/m1-cell-stub.XXXXXX")
trap 'rm -rf "$T"' EXIT
mkdir -p "$T/repo/bench" && cp -R "$SRC" "$T/repo/bench/m1"
printf '#!/usr/bin/env bash\necho "router stub $*" >> "$M1/router-calls"\nexit 1\n' > "$T/repo/bench/m1/router.sh"
CELL_SH=$T/repo/bench/m1/cell.sh
export M1=$T/m1 WORKSPACE=$T VENV=$T/venv HF_HOME=$T/hf
mkdir -p "$M1/data" && echo '[]' > "$M1/data/sharegpt.json"
fail=0
check() { if eval "$2"; then echo "ok    $1"; else echo "FAIL  $1"; fail=1; fi; }

"$CELL_SH" r1 sharegpt least_load 4 1 >/dev/null 2>&1; rc=$?
check "five arguments exit 2" '[ $rc -eq 2 ]'

"$CELL_SH" r1 sharegpt least_load 4 1 100 >/dev/null 2>&1; rc=$?
check "no current.env fails" '[ $rc -ne 0 ] && [ ! -d "$M1/results" ]'

printf 'CUR_MODEL_TAG=q35-9b\nCUR_PRESSURE=fits\nCUR_SIM=1\nCUR_BLOCK_TOKENS=16\nCUR_KV_TOKENS_A=1000\nCUR_KV_TOKENS_B=1000\n' > "$M1/current.env"
C=$M1/results/r1/q35-9b/fits/sharegpt/least_load-c4-s1

"$CELL_SH" r1 sharegpt magic 4 1 100 >/dev/null 2>&1; rc=$?
check "unknown policy fails before any output" '[ $rc -ne 0 ] && [ ! -d "$M1/results" ]'
"$CELL_SH" r1 w9 least_load 4 1 100 >/dev/null 2>&1; rc=$?
check "unknown workload fails" '[ $rc -ne 0 ] && [ ! -d "$M1/results" ]'
"$CELL_SH" r1 sharegpt least_load four 1 100 >/dev/null 2>&1; rc=$?
check "non-integer clients fails" '[ $rc -ne 0 ]'

mkdir -p "$C" && echo keep > "$C/stats.json" && touch "$C/DONE"
out=$("$CELL_SH" r1 sharegpt least_load 4 1 100 2>&1); rc=$?
check "DONE cell exits 0" '[ $rc -eq 0 ]'
check "DONE cell is reported as skipped" 'grep -q "skip (DONE)" <<< "$out"'
check "DONE cell is untouched and the router never started" '[ "$(cat "$C/stats.json")" = keep ] && [ ! -f "$M1/router-calls" ]'

rm "$C/DONE"
"$CELL_SH" r1 sharegpt least_load 4 1 100 >/dev/null 2>&1; rc=$?
check "failing cell exits non-zero" '[ $rc -ne 0 ]'
check "failing cell leaves no DONE" '[ ! -f "$C/DONE" ]'
check "partial cell was moved aside with its files" 'ls -d "$C".partial-* >/dev/null 2>&1 && [ "$(cat "$C".partial-*/stats.json)" = keep ]'
check "router was asked for the cell policy and directory" 'grep -qx "router stub least_load $C" "$M1/router-calls"'

[ "$fail" = 0 ] && echo "cell-stub: all checks passed"
exit "$fail"
