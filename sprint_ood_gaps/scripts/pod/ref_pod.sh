#!/bin/bash
mkdir -p /workspace/out; exec > /workspace/out/pipeline.log 2>&1; set -x
pip install -q --break-system-packages vllm==0.32.0 || { touch /workspace/out/FAILED; exit 1; }
cd /workspace && python ref_gen.py /workspace/prompts.jsonl /workspace/out/ref_replies.jsonl || { touch /workspace/out/FAILED; exit 1; }
touch /workspace/out/DONE
