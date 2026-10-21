#!/usr/bin/env bash
# Download the vendored vLLM multi-turn benchmark again and check every file against the
# sha256 table in vendor/VENDOR.md. Also checks the committed copies in vendor/multi_turn/.
# Usage: fetch-vendor.sh [--write]     --write replaces the committed copies with the download
set -euo pipefail
D=$(cd "$(dirname "$0")" && pwd)
TAG=v0.31.0
BASE=https://raw.githubusercontent.com/vllm-project/vllm/$TAG/benchmarks/multi_turn
TMP=$(mktemp -d "${TMPDIR:-/tmp}/m1-vendor.XXXXXX")
trap 'rm -rf "$TMP"' EXIT

sha256() { if command -v sha256sum >/dev/null; then sha256sum "$1" | cut -d' ' -f1; else shasum -a 256 "$1" | cut -d' ' -f1; fi; }

# Rows look like "| file | sha256 |"; the header and separator rows have no 64-hex column.
TABLE=$(awk -F'|' 'NF>=4 { f=$2; h=$3; gsub(/ /,"",f); gsub(/ /,"",h); if (h ~ /^[0-9a-f]{64}$/) print f, h }' "$D/vendor/VENDOR.md")
[ -n "$TABLE" ] || { echo "no hashes found in vendor/VENDOR.md" >&2; exit 1; }

bad=0
while read -r f want; do
  curl -sfL -o "$TMP/$f" "$BASE/$f" || { echo "download failed: $f" >&2; bad=1; continue; }
  got=$(sha256 "$TMP/$f")
  if [ "$got" = "$want" ]; then echo "ok        upstream $f"; else echo "MISMATCH  upstream $f ($got)"; bad=1; fi
  if [ -f "$D/vendor/multi_turn/$f" ]; then
    loc=$(sha256 "$D/vendor/multi_turn/$f")
    if [ "$loc" = "$want" ]; then echo "ok        local    $f"; else echo "MISMATCH  local    $f ($loc)"; bad=1; fi
  else
    echo "MISSING   local    $f"; bad=1
  fi
done <<< "$TABLE"

if [ "${1:-}" = --write ]; then
  [ "$bad" = 0 ] || { echo "not writing: upstream hashes do not all match VENDOR.md" >&2; exit 1; }
  mkdir -p "$D/vendor/multi_turn"
  while read -r f _; do cp "$TMP/$f" "$D/vendor/multi_turn/$f"; done <<< "$TABLE"
  echo "wrote $D/vendor/multi_turn"
fi
exit "$bad"
