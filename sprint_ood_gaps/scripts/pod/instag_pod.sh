#!/bin/bash
# Part 12 InsTag job: tag with InsTagger, upload to HF detectood/ood, then remove this pod. Hard limit 1.5 h.
mkdir -p /workspace/out; exec > /workspace/out/pipeline.log 2>&1; set -x
selfstop() { bash /workspace/self_remove.sh; }  # tested multi-method self-removal from step 9
upload() { python - <<'PY'
from huggingface_hub import HfApi
import os
tok = open("/workspace/token").read().strip()
HfApi(token=tok).upload_folder(folder_path="/workspace/out", repo_id="detectood/ood", repo_type="dataset",
    path_in_repo="results/raw/261029_instag_v3", commit_message="part 12 instag tags")
print("uploaded")
PY
}
timeout 5400 bash -c '
  pip install -q --break-system-packages vllm==0.32.0 huggingface_hub || exit 1
  cd /workspace && FORCE_FORMAT=A python instag_tag.py && touch /workspace/out/DONE
' || touch /workspace/out/FAILED
upload || { sleep 60; upload; }
rm -f /workspace/token
selfstop
