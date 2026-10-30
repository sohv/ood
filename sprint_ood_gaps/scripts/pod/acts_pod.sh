#!/bin/bash
mkdir -p /workspace/out; exec > /workspace/out/pipeline.log 2>&1; set -x
pip install -q --break-system-packages vllm==0.32.0 || { touch /workspace/out/FAILED; exit 1; }
cd /workspace && python acts_traffic.py || { touch /workspace/out/FAILED; exit 1; }
