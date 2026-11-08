#!/usr/bin/env bash
# Inject a spot reclaim in the shape ProcessFleet reads: stop file, SIGTERM to the slot's engine
# process, SIGKILL to its group after GRACE seconds. Returns when the group is gone.
# Usage: reclaim.sh SLOT GRACE
set -euo pipefail
SLOT=${1:?usage: reclaim.sh SLOT GRACE}
GRACE=${2:?usage: reclaim.sh SLOT GRACE}
source "$(dirname "$0")/common.sh"
PIDF=$M2_RUN_DIR/slot-$SLOT.pid
[ -f "$PIDF" ] || die "slot $SLOT has no process to reclaim"
PGID=$(cat "$PIDF")
touch "$M2_RUN_DIR/slot-$SLOT.stop"
kill -TERM "$PGID"
log "reclaim slot $SLOT: SIGTERM to $PGID, SIGKILL in ${GRACE}s"
T0=$SECONDS
while group_alive "$PGID"; do
  if (( SECONDS - T0 >= ${GRACE%.*} )); then
    kill -KILL -- "-$PGID" 2>/dev/null || true
    log "reclaim slot $SLOT: SIGKILL"
    break
  fi
  sleep 0.2
done
