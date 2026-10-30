#!/bin/bash
# runpod_watchdog.sh LIMIT_HOURS [NAME_PREFIX]: backstop for system cron on research-box (designs/step9_small.md).
# Deletes every Runpod pod that is running (desiredStatus RUNNING, or runtimeStatus error) and was created more than LIMIT_HOURS ago.
# With NAME_PREFIX, only pods whose name starts with it. Never touches stopped (EXITED) pods, e.g. a stopped pod kept for its volume.
# Pods whose id is a line in ~/.config/runpod/watchdog_keep are skipped. Uses the account key in ~/.config/runpod/env.
# Logs each deletion (and each listing failure) to $WATCHDOG_LOG (default ~/runpod_watchdog.log). DRY_RUN=1: log what it would delete.
# WATCHDOG_JSON=file: read the pod list from a file instead of the API (for testing).
set -u
LIMIT=${1:?usage: runpod_watchdog.sh LIMIT_HOURS [NAME_PREFIX]}; PREFIX=${2:-}
LOG=${WATCHDOG_LOG:-$HOME/runpod_watchdog.log}; KEEP=$HOME/.config/runpod/watchdog_keep
export PATH=/usr/local/bin:/usr/bin:/bin
source $HOME/.config/runpod/env
exec 9> /tmp/runpod_watchdog.$(id -u).lock; flock -n 9 || exit 0     # one instance at a time
ts() { date -u +%FT%TZ; }
if [ -n "${WATCHDOG_JSON:-}" ]; then LIST=$(cat "$WATCHDOG_JSON")
else LIST=$(runpodctl pod list -o json 2>&1) || { echo "$(ts) list failed: ${LIST:0:300}" >> $LOG; exit 1; }
fi
# one line per pod to delete: id name age_h cost_per_h
TARGETS=$(printf '%s' "$LIST" | python3 -c '
import json, sys, datetime as dt
limit, prefix, keep_file = float(sys.argv[1]), sys.argv[2], sys.argv[3]
try: keep = {l.strip() for l in open(keep_file) if l.strip()}
except FileNotFoundError: keep = set()
pods = json.load(sys.stdin); now = dt.datetime.now(dt.timezone.utc)
for p in pods if isinstance(pods, list) else pods.get("pods", []):
    running = p.get("desiredStatus") == "RUNNING" or p.get("runtimeStatus") == "error"
    if not running or p["id"] in keep or not p.get("name", "").startswith(prefix): continue
    age = (now - dt.datetime.fromisoformat(p["createdAt"].replace("Z", "+00:00"))).total_seconds() / 3600
    if age > limit: print(p["id"], p.get("name", "-").replace(" ", "_"), f"{age:.2f}", p.get("costPerHr", "?"))
' "$LIMIT" "$PREFIX" "$KEEP") || { echo "$(ts) parse failed" >> $LOG; exit 1; }
[ -n "$TARGETS" ] || exit 0
while read -r id name age cost; do
  if [ "${DRY_RUN:-0}" = 1 ]; then echo "$(ts) DRY_RUN would delete $id $name age ${age}h \$${cost}/h" >> $LOG; continue; fi
  out=$(runpodctl pod delete "$id" 2>&1); rc=$?
  echo "$(ts) deleted $id $name age ${age}h \$${cost}/h limit ${LIMIT}h rc=$rc ${out:0:200}" | tr '\n' ' ' >> $LOG; echo >> $LOG
done <<< "$TARGETS"
