#!/usr/bin/env bash
# Run locally: copy results, logs and manifests back from a pod.
# Usage: pull.sh HOST PORT
#   /workspace/m1/results/ -> bench/m1/results/, /workspace/m1/logs/ -> bench/m1/logs/,
#   manifests and the generated .env files (never the secrets file) -> bench/m1/results/manifest/.
set -euo pipefail
HOST=${1:?host}; PORT=${2:?ssh port}; KEY=${RUNPOD_SSH_KEY:-$HOME/.runpod/ssh/runpodctl-ssh-key}
D=$(cd "$(dirname "$0")" && pwd)
R=/workspace/m1
SSH=(ssh -i "$KEY" -p "$PORT" -o StrictHostKeyChecking=accept-new "root@$HOST")
mkdir -p "$D/results/manifest" "$D/logs"
"${SSH[@]}" "tar -C $R -czf - results" | tar -C "$D" -xzf -
"${SSH[@]}" "tar -C $R -czf - logs" | tar -C "$D" -xzf -
"${SSH[@]}" "cd $R && tar -czf - \$(ls env-manifest.txt pip-freeze.txt revisions.env knee.env current.env bootstrap.log bringup-*.env pressure-*.env data/*.sha256 2>/dev/null)" \
  | tar -C "$D/results/manifest" -xzf -
echo "pulled into $D/results and $D/logs"
