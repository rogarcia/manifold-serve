#!/usr/bin/env bash
# Run locally: copy the m0-redux scripts to a pod and (optionally) bootstrap it.
# Usage: push.sh HOST PORT [--bootstrap]     e.g. push.sh 81.27.69.178 31031 --bootstrap
# If a local .env exists next to this script it is copied too (mode 600).
set -euo pipefail
HOST=${1:?host}; PORT=${2:?ssh port}; KEY=${RUNPOD_SSH_KEY:-$HOME/.runpod/ssh/runpodctl-ssh-key}
D=$(cd "$(dirname "$0")" && pwd)
SSH=(ssh -i "$KEY" -p "$PORT" "root@$HOST")
"${SSH[@]}" 'mkdir -p /workspace/m0-redux/scripts/results'
scp -i "$KEY" -P "$PORT" -q "$D"/*.sh "root@$HOST:/workspace/m0-redux/scripts/"
scp -i "$KEY" -P "$PORT" -q "$D/results/pip-freeze.txt" "root@$HOST:/workspace/m0-redux/scripts/results/"
if [ -f "$D/.env" ]; then
  scp -i "$KEY" -P "$PORT" -q "$D/.env" "root@$HOST:/workspace/m0-redux/.env"
  "${SSH[@]}" 'chmod 600 /workspace/m0-redux/.env'
fi
[ "${3:-}" = --bootstrap ] && "${SSH[@]}" '/workspace/m0-redux/scripts/bootstrap.sh'
echo "pushed to $HOST:/workspace/m0-redux/scripts"
