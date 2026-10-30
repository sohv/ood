#!/bin/bash
# launch_small.sh: start step 9 small (designs/step9_small.md) on ONE Runpod GPU pod (B200, else H200). Run on research-box only after
# the owner says go. After this script returns, nothing on research-box is needed: the pod uploads its results to HF and removes itself.
# usage: bash scripts/pod/step9small/launch_small.sh        env: HARD_LIMIT (default 3h), UPLOAD_FEATURES (default 0), MIN_BALANCE (default 40)
set -uo pipefail
cd /home/cadcad/research/ood-gaps; source ~/.config/runpod/env
D=scripts/pod/step9small; RUN=runs/step9small; IN=$RUN/inputs; mkdir -p $RUN
HARD_LIMIT=${HARD_LIMIT:-3h}; UPLOAD_FEATURES=${UPLOAD_FEATURES:-0}; MIN_BALANCE=${MIN_BALANCE:-40}
NAME=step9small-$(date -u +%m%d%H%M); IMAGE=runpod/pytorch:1.0.3-cu1281-torch291-ubuntu2404
die() { echo "STOP: $*"; exit 1; }

# 0. backstop must be in place before any pod exists
crontab -l 2>/dev/null | grep -q "step9small/runpod_watchdog.sh" || die "watchdog cron line not installed (designs/step9_small.md, checklist)"
# 1. inputs (deterministic; about 35 s)
[ -f $IN/manifest.json ] || .venv/bin/python -I $D/prepare_small.py $IN > $RUN/prepare.log 2>&1 || die "prepare_small.py failed, see $RUN/prepare.log"
# 2. balance
BAL=$(runpodctl user -o json 2>/dev/null | python3 -c "import json,sys;d=json.load(sys.stdin);print(d.get('clientBalance', d.get('balance', '')))")
echo "Runpod balance before: $BAL"
python3 -c "import sys; sys.exit(0 if float('${BAL:-0}' or 0) >= $MIN_BALANCE else 1)" || die "balance $BAL below $MIN_BALANCE"
# 3. config and a token file holding HF_TOKEN2 only (mode 600, deleted when this script exits; on the pod it is deleted after the last upload)
cat > $RUN/config.env <<EOF
HARD_LIMIT=$HARD_LIMIT
UPLOAD_FEATURES=$UPLOAD_FEATURES
HF_REPO=detectood/ood
HF_PATH=results/raw/261029_step9_small_v1
EOF
TOK=$(mktemp -d); chmod 700 $TOK; trap 'rm -rf "$TOK"' EXIT
( umask 077; sed -n 's/^HF_TOKEN2=//p' /home/cadcad/research/.env | tr -d '"'"'"' \r' > $TOK/hf_token )
[ -s $TOK/hf_token ] || die "HF_TOKEN2 not found in /home/cadcad/research/.env"
# 4. preflight upload with the same code and token: config + inputs manifest -> HF_PATH/preflight/
mkdir -p $TOK/ws/out; cp $RUN/config.env $TOK/ws/; cp $TOK/hf_token $TOK/ws/; cp $IN/manifest.json $IN/arm_sizes.json $RUN/config.env $TOK/ws/out/
WS=$TOK/ws .venv/bin/python -I $D/upload.py preflight || die "preflight upload to HF failed"
# 5. one pod: B200, else H200. launch.py creates it, waits for ssh, copies files to /workspace, starts job.sh detached (setsid nohup).
EXTRA=(scripts/pod/acts_dolci.py scripts/pod/features_sohan.py $D/run.sh $D/gate1.py $D/search_small.py $D/upload.py $D/self_remove.sh
       $RUN/config.env $TOK/hf_token $IN/queries.jsonl $IN/q_l20_existing.npy $IN/sub_idx.npy $IN/gate2_sub.npz $IN/arms_sub.npz)
J=runs/misbehave100k/launch/$NAME.json; ok=0
for g in "NVIDIA B200" "NVIDIA H200"; do
  echo "trying $g"; .venv/bin/python -I scripts/pod/launch.py $NAME $IMAGE $IN/dolci_sub.jsonl "$g" $D/job.sh "${EXTRA[@]}" >> $RUN/launch.log 2>&1 && { ok=1; break; }
  ID=$(python3 -c "import json;print(json.load(open('$J')).get('id',''))" 2>/dev/null)
  if [ -n "$ID" ]; then   # created but no ssh / copy failed: remove it now rather than leave it idle
    runpodctl pod delete $ID; die "pod $ID on $g created but launch failed after creation; deleted; see $RUN/launch.log"
  fi
done
[ $ok = 1 ] || die "no B200 or H200 could be created; see $RUN/launch.log"
read IP PORT ID < <(python3 -c "import json;d=json.load(open('$J'));print(*d['host'],d['id'])")
# 6. pod id on the pod, a fallback in case RUNPOD_POD_ID is not visible to the job (read only at self-removal)
ssh -o ConnectTimeout=20 -p $PORT root@$IP "echo $ID > /workspace/pod_id.txt" || echo "warning: could not write pod_id.txt"
GPU=$(python3 -c "import json;print(json.load(open('$J'))['gpu'])")
echo "started $NAME pod $ID ($GPU) at $(date -u +%FT%TZ); results -> hf://datasets/detectood/ood/results/raw/261029_step9_small_v1/"
echo "ledger row: | $(TZ=America/Denver date +'%d %b %H:%M') | ood-gaps | step 9 small (designs/step9_small.md): Olmo L8/20/28 on 20% Dolci + 49,629 queries, masked kNN | Runpod $GPU $ID; balance \$$BAL before; expected ~1-1.5 h, ~\$8-12; hard limit $HARD_LIMIT, self-removing | running |"
