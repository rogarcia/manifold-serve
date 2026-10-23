#!/usr/bin/env bash
# Run locally after push.sh: copy the state that the Modal calibration produced onto a pod, so the
# matrix runs there with the same knee, working set and pool sizes.
# Usage: seed-pod.sh HOST PORT [SRC]     SRC defaults to bench/m1/modal/results (from modal/pull.sh)
#   SRC/manifest/knee.env and pressure-q35-9b.env -> /workspace/m1/
#   SRC/calib/ -> /workspace/m1/results/calib-modal/ (control-model.sh sizes the q3-8b pool from it)
# Refuses to overwrite a knee.env or pressure file already on the pod.
set -euo pipefail
HOST=${1:?host}; PORT=${2:?ssh port}; KEY=${RUNPOD_SSH_KEY:-$HOME/.runpod/ssh/runpodctl-ssh-key}
D=$(cd "$(dirname "$0")" && pwd)
SRC=${3:-$D/modal/results}
R=/workspace/m1
SSH=(ssh -i "$KEY" -p "$PORT" -o StrictHostKeyChecking=accept-new "root@$HOST")

for f in knee.env pressure-q35-9b.env; do
  [ -s "$SRC/manifest/$f" ] || { echo "missing $SRC/manifest/$f: run bench/m1/modal/pull.sh first" >&2; exit 1; }
done
[ -d "$SRC/calib/q35-9b/nocap" ] || { echo "missing $SRC/calib/q35-9b/nocap" >&2; exit 1; }

"${SSH[@]}" "mkdir -p $R/results && for f in knee.env pressure-q35-9b.env; do
  if [ -e $R/\$f ]; then echo \"$R/\$f already exists; not overwriting\" >&2; exit 1; fi; done"
for f in knee.env pressure-q35-9b.env; do
  "${SSH[@]}" "cat > $R/$f" < "$SRC/manifest/$f"
done
# Finished cells only; .partial-* directories are runs that were moved aside.
COPYFILE_DISABLE=1 tar -C "$SRC/calib" --no-xattrs --no-fflags --no-mac-metadata \
  --uid 0 --gid 0 --uname root --gname root --exclude='*.partial-*' -czf - . \
  | "${SSH[@]}" "mkdir -p $R/results/calib-modal && tar -C $R/results/calib-modal --no-same-owner -xzf -"
"${SSH[@]}" "cat $R/knee.env $R/pressure-q35-9b.env; ls $R/results/calib-modal/q35-9b/nocap/sharegpt | wc -l"
echo "seeded $HOST:$R from $SRC"
