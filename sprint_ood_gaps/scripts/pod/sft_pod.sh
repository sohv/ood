#!/bin/bash
# runs on the pod: Sohan's run_rollouts for Olmo-3-7B-Instruct-SFT (designs/dpo_vs_sft.md), same sampling as the DPO replies
mkdir -p /workspace/out; exec > /workspace/out/pipeline.log 2>&1; set -x
date -u +"start %FT%TZ"
cd /workspace && tar xzf ood_repo.tgz || { touch /workspace/out/FAILED; exit 1; }
pip install -q --break-system-packages vllm==0.32.0 simple_parsing || { touch /workspace/out/FAILED; exit 1; }
date -u +"installed %FT%TZ"
git config --global --add safe.directory "*"
cd /workspace/ood; export VLLM_USE_FLASHINFER_SAMPLER=0
python -m scripts.run_rollouts --pool_path /workspace/prompts.jsonl --run_dir /workspace/out/rollouts --model_id allenai/Olmo-3-7B-Instruct-SFT --model_revision e1452fc572d51966ff4aaeb25118b891eb93e549 --n_rollouts 1 --temperature 0.6 --top_p 0.95 \
  --max_new_tokens 2048 --max_model_len 8192 --chunk 500 --seed 0 --resume true || { touch /workspace/out/FAILED; exit 1; }
date -u +"done %FT%TZ"; touch /workspace/out/DONE
