#!/bin/bash
# test_self_remove_cpu.sh: OPTIONAL paid test (a few cents) of the one unverified link, self-removal from inside a pod. Creates one CPU pod
# with the same image as the GPU run, copies self_remove.sh, runs it detached, and checks from research-box that the pod disappears within
# 6 min. If it does not, deletes the pod from here and reports FAIL (then rely on the watchdog, or fix self_remove.sh before go).
# usage: bash scripts/pod/step9small/test_self_remove_cpu.sh     (research-box; needs the owner's go like any paid run)
set -uo pipefail
cd /home/cadcad/research/ood-gaps; source ~/.config/runpod/env
IMAGE=runpod/pytorch:1.0.3-cu1281-torch291-ubuntu2404; NAME=step9small-selftest; KEY=~/.ssh/id_ed25519
OUT=$(runpodctl pod create --compute-type CPU --image $IMAGE --name $NAME --ports 22/tcp --container-disk-in-gb 20 2>&1)
ID=$(printf '%s' "$OUT" | python3 -c "import json,sys;s=sys.stdin.read();print(json.loads(s[s.index('{'):])['id'])" 2>/dev/null)
[ -n "$ID" ] || { echo "create failed: ${OUT:0:500}"; exit 1; }
echo "created CPU pod $ID $(date -u +%T)"; trap 'runpodctl pod get $ID > /dev/null 2>&1 && runpodctl pod delete $ID > /dev/null 2>&1 && echo "deleted $ID from research-box"' EXIT
for i in $(seq 1 80); do   # up to 20 min for ssh (image pull)
  read IP PORT < <(runpodctl pod get $ID 2>/dev/null | python3 -c "import json,sys;d=json.load(sys.stdin);s=d.get('ssh') or {};print(s.get('ip',''),s.get('port',''))")
  [ -n "${IP:-}" ] && ssh -i $KEY -o ConnectTimeout=10 -o StrictHostKeyChecking=accept-new -o BatchMode=yes -p $PORT root@$IP true 2>/dev/null && break
  IP=""; sleep 15
done
[ -n "${IP:-}" ] || { echo "FAIL: no ssh in 20 min"; exit 1; }
echo "ssh up $(date -u +%T)"
scp -i $KEY -P $PORT -o StrictHostKeyChecking=accept-new scripts/pod/step9small/self_remove.sh root@$IP:/workspace/ > /dev/null
ssh -i $KEY -p $PORT root@$IP "echo $ID > /workspace/pod_id.txt; cd /workspace && (setsid nohup bash self_remove.sh > /workspace/self_remove.log 2>&1 < /dev/null &)"
for i in $(seq 1 24); do
  sleep 15
  st=$(runpodctl pod get $ID 2>/dev/null | python3 -c "import json,sys;print(json.load(sys.stdin).get('desiredStatus') or 'GONE')" 2>/dev/null || echo GONE)
  case "$st" in GONE|TERMINATED) echo "PASS: pod $ID removed itself after $((i * 15)) s"; trap - EXIT; exit 0;;
                EXITED) echo "PARTIAL: pod $ID stopped itself but was not removed (status EXITED)"; exit 1;; esac
  [ $i = 2 ] && ssh -i $KEY -o ConnectTimeout=10 -p $PORT root@$IP "cat /workspace/self_remove.log" 2>/dev/null | head -12
done
echo "FAIL: pod $ID still listed after 6 min; log:"; ssh -i $KEY -o ConnectTimeout=10 -p $PORT root@$IP "cat /workspace/self_remove.log" 2>/dev/null | tail -20
exit 1
