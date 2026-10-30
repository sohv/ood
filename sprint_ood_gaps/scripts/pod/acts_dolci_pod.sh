#!/bin/bash
mkdir -p /workspace/out; exec > /workspace/out/pipeline.log 2>&1; set -x
pip install -q --break-system-packages vllm==0.32.0 || { touch /workspace/out/FAILED; exit 1; }
python -c "import torch, transformers, numpy; print('versions', torch.__version__, transformers.__version__, numpy.__version__, torch.cuda.get_device_name(0))" > /workspace/out/versions.txt
cd /workspace && python acts_dolci.py || { touch /workspace/out/FAILED; exit 1; }
