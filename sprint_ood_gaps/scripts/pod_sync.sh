#!/usr/bin/env bash
# mirror the pod's /workspace/out (rollouts, logs, adapters, step markers) into runs/pod_sync/<pod>/ every INTERVAL s; adapted from nla-vs-lenses/scripts/pod_sync.sh
# usage: pod_sync.sh HOST PORT PODNAME [INTERVAL=120|once] [END_MARKER=DONE]. Run with `once` before any runpodctl pod stop|delete.
H=$1; PT=$2; NAME=$3; INT=${4:-120}; END=${5:-DONE}; DEST=$(dirname "$0")/../runs/pod_sync/$NAME; mkdir -p "$DEST"
sync_once() {
  rsync -az --timeout=60 -e "ssh -o ConnectTimeout=20 -o StrictHostKeyChecking=accept-new -p $PT" \
    root@$H:/workspace/out/ "$DEST"/ 2>>"$DEST/sync.err" && date -u +"synced %FT%TZ" >> "$DEST/sync.log"
}
if [ "$INT" = once ]; then sync_once; exit $?; fi
fails=0
while true; do
  if sync_once; then fails=0; else fails=$((fails+1)); [ $fails -ge 5 ] && { echo "pod unreachable 5x, exiting $(date -u)" >> "$DEST/sync.log"; exit 0; }; fi
  [ -f "$DEST/$END" ] || [ -f "$DEST/FAILED" ] || [ -f "$DEST/FAILED_guard" ] && { echo "end marker seen $(date -u)" >> "$DEST/sync.log"; exit 0; }
  sleep "$INT"
done
