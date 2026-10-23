#!/usr/bin/env bash
# Run locally: copy results, logs and manifests from the Modal Volume into bench/m1/modal/,
# the Modal counterpart of pull.sh. Kept apart from bench/m1/results/ so RunPod and Modal runs
# never mix.
# Usage: bench/m1/modal/pull.sh     M1_VOLUME (default manifold-m1) names the Volume.
#   m1/results/ -> bench/m1/modal/results/, m1/logs/ -> bench/m1/modal/logs/, manifests and the
#   generated .env files -> bench/m1/modal/results/manifest/.
set -euo pipefail
D=$(cd "$(dirname "$0")" && pwd)
VOL=${M1_VOLUME:-manifold-m1}
MODAL=(uv run --no-sync modal)
mkdir -p "$D/results/manifest" "$D/logs"
for dir in results logs; do
  "${MODAL[@]}" volume get --force "$VOL" "m1/$dir" "$D/" || echo "not on the Volume yet: m1/$dir"
done
for f in env-manifest.txt pip-freeze.txt revisions.env knee.env bringup-q35-9b.env bringup-q3-8b.env \
         pressure-q35-9b.env pressure-q3-8b.env data/sharegpt.json.sha256 data/pg1184.txt.sha256; do
  "${MODAL[@]}" volume get --force "$VOL" "m1/$f" "$D/results/manifest/$(basename "$f")" >/dev/null 2>&1 \
    || echo "not on the Volume yet: m1/$f"
done
echo "pulled into $D/results and $D/logs"
