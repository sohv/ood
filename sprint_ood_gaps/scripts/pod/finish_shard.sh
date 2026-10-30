#!/bin/bash
# finish_shard.sh K LAUNCHNAME: final sync, verify one reply per prompt, delete the pod only if complete, then grade.
K=$1; L=$2; cd /home/cadcad/research/ood-gaps; source ~/.config/runpod/env
read IP PORT ID < <(python3 -c "import json;d=json.load(open('runs/misbehave100k/launch/$L.json'));print(*d['host'],d['id'])")
bash scripts/pod_sync.sh $IP $PORT mb100k-shard$K once || { echo "shard$K final sync failed, pod kept"; exit 1; }
.venv/bin/python -c "
import json,sys
ids=[json.loads(l)['prompt_id'] for l in open('runs/pod_sync/mb100k-shard$K/rollouts/rollouts.jsonl')]
want={json.loads(l)['id'] for l in open('runs/misbehave100k/shards/shard$K.jsonl')}
print('shard$K', len(ids), len(set(ids)), set(ids)==want); sys.exit(0 if len(ids)==len(set(ids)) and set(ids)==want else 1)" || { echo "shard$K incomplete, pod kept"; exit 1; }
runpodctl pod delete $ID > /dev/null 2>&1 && echo "shard$K pod $ID deleted"
.venv/bin/python scripts/misbehave100k_grade_shard.py runs/misbehave100k/shards/shard$K.jsonl runs/pod_sync/mb100k-shard$K/rollouts/rollouts.jsonl runs/misbehave100k/graded_shard$K 2> runs/misbehave100k/graded_shard$K.err | tail -20
