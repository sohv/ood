#!/bin/bash
# self_remove.sh: remove (terminate) this Runpod pod from inside it. Exit codes of the CLI are not trusted: if a removal works, the container
# is killed and nothing after it runs, so each method is followed by a 30 s wait and the next method is tried only if we are still alive.
# Credentials: only the pod-scoped RUNPOD_API_KEY that Runpod injects into the container (not the account key). Not verified before
# launch (see designs/step9_small.md); the research-box cron watchdog is the backstop. DRY_RUN=1 prints the commands without running them.
W=${WS:-/workspace}; DRY=${DRY_RUN:-0}
# Runpod sets these in PID 1's environment; shells started over SSH may not inherit them
[ -f /etc/rp_environment ] && source /etc/rp_environment
for v in RUNPOD_POD_ID RUNPOD_API_KEY; do
  [ -n "${!v:-}" ] || export $v="$(tr '\0' '\n' 2>/dev/null < /proc/1/environ | sed -n "s/^$v=//p" | head -1)"
done
[ -n "${RUNPOD_POD_ID:-}" ] || RUNPOD_POD_ID=$(cat $W/pod_id.txt 2>/dev/null)
echo "self_remove $(date -u +%FT%TZ): pod ${RUNPOD_POD_ID:-UNKNOWN}, pod key $([ -n "${RUNPOD_API_KEY:-}" ] && echo present || echo ABSENT), runpodctl $(command -v runpodctl > /dev/null && runpodctl version 2>&1 | head -1 || echo ABSENT)"
[ -n "${RUNPOD_POD_ID:-}" ] || { echo "no pod id; cannot self-remove"; exit 1; }
try() { echo "+ $*" | sed "s/${RUNPOD_API_KEY:-NOKEY}/<key>/g"; [ "$DRY" = 1 ] && return 0; "$@" 2>&1 | tail -3; sleep 30; }
gql='{"query":"mutation { podTerminate(input: {podId: \"'"$RUNPOD_POD_ID"'\"}) }"}'
for round in 1 2 3 4 5 6 7 8; do
  echo "round $round"
  if command -v runpodctl > /dev/null; then
    try runpodctl remove pod "$RUNPOD_POD_ID"          # syntax of the runpodctl 1.x preinstalled in pods (deprecated alias in 3.x)
    try runpodctl pod delete "$RUNPOD_POD_ID"          # runpodctl 3.x syntax
  fi
  try curl -s -m 30 -X DELETE "https://rest.runpod.io/v1/pods/$RUNPOD_POD_ID" -H "Authorization: Bearer $RUNPOD_API_KEY"
  try curl -s -m 30 -X POST "https://api.runpod.io/graphql" -H "Content-Type: application/json" -H "Authorization: Bearer $RUNPOD_API_KEY" -d "$gql"
  [ "$DRY" = 1 ] && break
  sleep 60
done
# still alive after ~25 min of attempts: stop instead (stops GPU billing; container disk is kept until the watchdog deletes the pod)
echo "removal failed; trying stop"
command -v runpodctl > /dev/null && { try runpodctl stop pod "$RUNPOD_POD_ID"; try runpodctl pod stop "$RUNPOD_POD_ID"; }
try curl -s -m 30 -X POST "https://rest.runpod.io/v1/pods/$RUNPOD_POD_ID/stop" -H "Authorization: Bearer $RUNPOD_API_KEY"
exit 1
