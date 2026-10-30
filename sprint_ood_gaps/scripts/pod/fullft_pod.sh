#!/bin/bash
# full-FT re-test: train one arm (ARM file /workspace/arm.txt: fix or control), then answer fix2a's held-out eval prompts.
mkdir -p /workspace/out; exec > /workspace/out/pipeline.log 2>&1; set -x
ARM=$(cat /workspace/arm.txt)
cd /workspace && tar xzf ood_repo.tgz || { touch /workspace/out/FAILED; exit 1; }
pip install -q --break-system-packages vllm==0.32.0 simple_parsing accelerate || { touch /workspace/out/FAILED; exit 1; }
git config --global --add safe.directory "*"; cp /workspace/fullft_train.py /workspace/ood/scripts/; cd /workspace/ood
mkdir -p /workspace/train && cp /workspace/train_$ARM.jsonl /workspace/train/
python -m scripts.fullft_train --train_dir /workspace/train --run_dir /workspace/dry_run --adapters $ARM --seed 0 --num_tasks 16 --epochs 1 || { touch /workspace/out/FAILED_dry; exit 1; }
rm -rf /workspace/dry_run; touch /workspace/out/STEP_dry_ok
python -m scripts.fullft_train --train_dir /workspace/train --run_dir /workspace/train_run --adapters $ARM --seed 0 || { touch /workspace/out/FAILED; exit 1; }
cp /workspace/train_run/adapters/*_loss.jsonl /workspace/out/ 2>/dev/null; touch /workspace/out/STEP_trained
python /workspace/gen_local.py /workspace/train_run/adapters/$ARM /workspace/eval_pool.jsonl /workspace/out/eval_$ARM.jsonl || { touch /workspace/out/FAILED; exit 1; }
cp /workspace/train_run/adapters/*_loss.jsonl /workspace/train_run/*.json /workspace/train_run/*.log /workspace/out/ 2>/dev/null   # model weights stay on the pod (not synced)
touch /workspace/out/DONE
