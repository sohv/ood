#!/bin/bash
# full-FT re-test: Olmo-3.1-32B-Instruct teacher replies (fix2a teacher settings, max_new_tokens raised 1024 -> 4096), then
# Olmo-3-7B-Instruct-DPO replies on the selection prompts if select_pool.jsonl is present.
mkdir -p /workspace/out; exec > /workspace/out/pipeline.log 2>&1; set -x
cd /workspace && tar xzf ood_repo.tgz || { touch /workspace/out/FAILED; exit 1; }
pip install -q --break-system-packages vllm==0.32.0 simple_parsing || { touch /workspace/out/FAILED; exit 1; }
git config --global --add safe.directory "*"; cd /workspace/ood; export VLLM_USE_FLASHINFER_SAMPLER=0
python -m scripts.run_rollouts --pool_path /workspace/prompts.jsonl --run_dir /workspace/out/teacher --model_id allenai/Olmo-3.1-32B-Instruct \
  --model_revision ac0587e4a7744a551c059d8cd17ba220bc940dae --n_rollouts 1 --temperature 0.6 --top_p 0.95 --max_new_tokens 4096 --max_model_len 8192 \
  --chunk 600 --seed 0 --resume true || { touch /workspace/out/FAILED; exit 1; }
if [ -f /workspace/select_pool.jsonl ]; then
python -m scripts.run_rollouts --pool_path /workspace/select_pool.jsonl --run_dir /workspace/out/base_select --n_rollouts 1 --temperature 0.6 --top_p 0.95 \
  --max_new_tokens 4096 --max_model_len 8192 --chunk 600 --seed 0 --resume true || { touch /workspace/out/FAILED; exit 1; }
fi
touch /workspace/out/DONE
