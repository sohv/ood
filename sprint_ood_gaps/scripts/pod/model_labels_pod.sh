#!/bin/bash
# Part 11 job: generate, score under SFT and base, upload to HF detectood/ood, then remove this pod. Hard limit 2.5 h.
mkdir -p /workspace/out; exec > /workspace/out/pipeline.log 2>&1; set -x
selfstop() { runpodctl remove pod "$RUNPOD_POD_ID" || runpodctl stop pod "$RUNPOD_POD_ID" || kill -TERM 1; }
upload() { python - <<'PY'
from huggingface_hub import HfApi
import os
tok = open("/workspace/token").read().strip()
HfApi(token=tok).upload_folder(folder_path="/workspace/out", repo_id="detectood/ood", repo_type="dataset",
    path_in_repo="results/raw/261029_model_labels_v1", commit_message="part 11 model labels")
print("uploaded")
PY
}
timeout 9000 bash -c '
  pip install -q --break-system-packages vllm==0.32.0 huggingface_hub || exit 1
  cd /workspace && python model_labels_gen.py && python model_labels_score.py sft && python model_labels_score.py base && touch /workspace/out/DONE
' || touch /workspace/out/FAILED
upload || { sleep 60; upload; }
rm -f /workspace/token
selfstop
