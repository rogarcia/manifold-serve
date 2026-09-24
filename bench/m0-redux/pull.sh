#!/usr/bin/env bash
# Run locally after each driver finishes: copy results, logs and manifests back.
# Usage: pull.sh HOST PORT
set -euo pipefail
HOST=${1:?host}; PORT=${2:?ssh port}; KEY=${RUNPOD_SSH_KEY:-$HOME/.runpod/ssh/runpodctl-ssh-key}
D=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$D/results" "$D/logs"
scp -i "$KEY" -P "$PORT" -q -r "root@$HOST:/workspace/m0-redux/results/." "$D/results/"
scp -i "$KEY" -P "$PORT" -q "root@$HOST:/workspace/m0-redux/logs/*.log" "$D/logs/"
scp -i "$KEY" -P "$PORT" -q "root@$HOST:/workspace/m0-redux/env-manifest.txt" "root@$HOST:/workspace/m0-redux/pip-freeze.txt" "$D/results/"
echo "pulled into $D/{results,logs}"
