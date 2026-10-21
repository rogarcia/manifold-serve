#!/usr/bin/env bash
# Run locally: copy the repo working tree to /workspace/m1/repo on a pod, and optionally bootstrap.
# Usage: push.sh HOST PORT [--bootstrap]     e.g. push.sh 81.27.69.178 31031 --bootstrap
# Sends the working tree as it is (uncommitted edits included), without .git, virtualenvs,
# results, logs or .scratch, and writes the commit it came from to repo/GIT_SHA.
# bench/m1/.env, if present, goes to /workspace/m1/.env (mode 600).
set -euo pipefail
HOST=${1:?host}; PORT=${2:?ssh port}; KEY=${RUNPOD_SSH_KEY:-$HOME/.runpod/ssh/runpodctl-ssh-key}
D=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$D/../.." && pwd)
R=/workspace/m1
SSH=(ssh -i "$KEY" -p "$PORT" -o StrictHostKeyChecking=accept-new "root@$HOST")

SHA=$(git -C "$REPO" rev-parse HEAD)
[ -z "$(git -C "$REPO" status --porcelain)" ] || SHA="$SHA-dirty"
"${SSH[@]}" "mkdir -p $R/repo"
# COPYFILE_DISABLE and --no-xattrs keep macOS metadata out of the archive.
COPYFILE_DISABLE=1 tar -C "$REPO" --no-xattrs \
  --exclude=.git --exclude=.venv --exclude=venv --exclude=results --exclude=logs \
  --exclude=.scratch --exclude=__pycache__ --exclude=.pytest_cache --exclude=.ruff_cache --exclude=.env \
  -czf - . | "${SSH[@]}" "tar -C $R/repo -xzf - && echo $SHA > $R/repo/GIT_SHA"
if [ -f "$D/.env" ]; then
  "${SSH[@]}" "umask 077 && cat > $R/.env" < "$D/.env"
fi
echo "pushed $SHA to $HOST:$R/repo"
if [ "${3:-}" = --bootstrap ]; then "${SSH[@]}" "$R/repo/bench/m1/bootstrap.sh 2>&1 | tee -a $R/bootstrap.log"; fi
