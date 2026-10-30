#!/bin/bash
# run.sh: step 9 small pipeline on the pod, called by job.sh under the hard time limit. Any failure exits at once (job.sh then uploads
# the logs and removes the pod). Steps and their UTC start times go to /workspace/out/steps.txt.
set -x
W=${WS:-/workspace}; cd $W; source $W/config.env
step() { echo "$1 $(date -u +%FT%TZ)" >> $W/out/steps.txt; }
fail() { touch $W/out/FAILED_$1; python upload.py results; exit 1; }
[ -f dolci_sub.jsonl ] || mv prompts.jsonl dolci_sub.jsonl   # launch.py copies the Dolci subsample to prompts.jsonl

step host   # host check (2026-10-21 audit: a cgroup CPU quota below nproc throttled a shard 2x)
{ echo "nproc $(nproc)"; echo "cpu.max $(cat /sys/fs/cgroup/cpu.max 2>/dev/null)"; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
  df -h $W | tail -1; free -g | head -2; echo "pod ${RUNPOD_POD_ID:-?}"; } > $W/out/host.txt 2>&1
# can this pod remove itself? log names (not values) of the RUNPOD_* variables, runpodctl version, and a read of this pod with the pod key
{ set +x; } 2>/dev/null   # no tracing here: the export below would print the pod key into job.log, which is uploaded
while IFS='=' read -r k v; do [ -n "$k" ] && [ -z "${!k:-}" ] && export "$k=$v"; done < <(tr '\0' '\n' 2>/dev/null < /proc/1/environ | grep '^RUNPOD_')
[ -n "${RUNPOD_POD_ID:-}" ] || RUNPOD_POD_ID=$(cat $W/pod_id.txt 2>/dev/null)
{ echo "RUNPOD vars: $(env | grep -o '^RUNPOD_[A-Z_]*' | sort | tr '\n' ' ')"; echo "runpodctl: $(command -v runpodctl) $(runpodctl version 2>&1 | head -1)"
  { runpodctl get pod "$RUNPOD_POD_ID" || runpodctl pod get "$RUNPOD_POD_ID"; } > /dev/null 2>&1; echo "pod-key read of this pod rc=$?"; } >> $W/out/host.txt 2>&1
set -x
Q=$(awk '$1 != "max" {print int($1 / $2)}' /sys/fs/cgroup/cpu.max 2>/dev/null); export OMP_NUM_THREADS=${Q:-$(nproc)}
[ "${OMP_NUM_THREADS:-0}" -ge 1 ] || export OMP_NUM_THREADS=8

step pip    # same install as last night's acts_dolci_pod.sh and the traffic run (it moves torch to 2.13.0+cu130; gate 1 checks the match)
pip install -q --break-system-packages vllm==0.32.0 || fail pip
python -c "import torch, transformers, numpy, huggingface_hub; print('versions', torch.__version__, transformers.__version__, numpy.__version__, huggingface_hub.__version__, torch.cuda.get_device_name(0))" > $W/out/versions.txt
pip freeze > $W/out/pip_freeze.txt 2>/dev/null

step upload_early   # token and path checked from the pod before any GPU work; no point computing what cannot be saved
python upload.py early || { touch $W/out/FAILED_upload_early; exit 1; }

extract() {   # $1 prompts file, $2 output dir. acts_dolci.py unchanged (Sohan's extractor, layers 8/20/28, content mean, last 512 tokens)
  cp $1 $W/prompts.jsonl; rm -f $W/out/DONE
  python acts_dolci.py > $W/out/extract_$(basename $2).log 2>&1 || return 1
  [ -f $W/out/DONE ] || return 1
  mkdir -p $2 && mv $W/out/acts_*.npz $2/ && rm -f $W/out/DONE
}
step extract_queries
extract $W/queries.jsonl $W/feat_queries || fail extract_queries
step gate1
python gate1.py || fail gate1
step extract_dolci
extract $W/dolci_sub.jsonl $W/feat_dolci || fail extract_dolci
step search
python search_small.py > $W/out/search.log 2>&1 || fail search
step upload_results
python upload.py results || { touch $W/out/FAILED_upload_results; exit 1; }
if [ "${UPLOAD_FEATURES:-0}" = 1 ]; then step upload_features; python upload.py features || touch $W/out/FAILED_upload_features; fi
step done; touch $W/out/ALL_DONE
