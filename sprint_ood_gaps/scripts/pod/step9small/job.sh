#!/bin/bash
# job.sh: entry point on the pod (started detached by scripts/pod/launch.py). Does not depend on any session on research-box.
# 1) starts an independent guard that removes the pod at HARD_LIMIT + 45 min whatever else happens;
# 2) runs run.sh under a hard `timeout` (HARD_LIMIT, default 3h; kills run.sh's whole process group);
# 3) uploads whatever is in /workspace/out (bounded to 15 min), deletes the token file;
# 4) removes the pod (self_remove.sh). Design: designs/step9_small.md.
W=${WS:-/workspace}; mkdir -p $W/out
exec >> $W/out/job.log 2>&1
source $W/config.env
echo "job start $(date -u +%FT%TZ) hard limit $HARD_LIMIT"

# independent guard: a detached sleeper in its own session, so it survives job.sh being killed
LIM_S=$(python3 -c "import re,sys;s='$HARD_LIMIT';m=re.fullmatch(r'(\d+)([smh]?)',s);print(int(m[1])*{'':1,'s':1,'m':60,'h':3600}[m[2]])")
GUARD_S=$((LIM_S + ${GUARD_MARGIN_S:-2700}))
setsid nohup bash -c "sleep $GUARD_S; echo 'guard fired' >> $W/out/job.log; bash $W/self_remove.sh >> $W/out/job.log 2>&1" > /dev/null 2>&1 < /dev/null &
echo "guard pid $! fires after ${GUARD_S}s"

timeout --signal=TERM --kill-after=120 "$HARD_LIMIT" bash $W/run.sh
rc=$?
echo "run.sh exit $rc at $(date -u +%FT%TZ) (124 or 137 = hit the hard limit)"
echo $rc > $W/out/run_exit_code.txt

timeout 15m python $W/upload.py final || echo "final upload failed or timed out"
shred -u $W/hf_token 2>/dev/null || rm -f $W/hf_token
bash $W/self_remove.sh
