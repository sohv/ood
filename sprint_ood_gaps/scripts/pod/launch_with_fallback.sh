#!/bin/bash
# launch one extraction shard, trying GPU types in order until a pod is created; usage: launch_with_fallback.sh SHARD
S=$1; cd "$(dirname "$0")/../.."; source ~/.config/runpod/env
for g in "NVIDIA B200" "NVIDIA H200" "NVIDIA H100 80GB HBM3" "NVIDIA H100 NVL" "NVIDIA H100 PCIe" "NVIDIA A100-SXM4-80GB" "NVIDIA A100 80GB PCIe"; do
  .venv/bin/python -I scripts/pod/launch.py acts-dolci-$S runpod/pytorch:1.0.3-cu1281-torch291-ubuntu2404 runs/acts_dolci/shard$S.jsonl "$g" scripts/pod/acts_dolci_pod.sh scripts/pod/acts_dolci.py scripts/pod/features_sohan.py > runs/acts_dolci/launch_$S.log 2>&1 && { echo "shard $S on $g"; exit 0; }
  grep -q '"id"' runs/misbehave100k/launch/acts-dolci-$S.json 2>/dev/null && { echo "shard $S created but failed later on $g; stopping fallback"; exit 1; }
done; echo "shard $S: no GPU available"; exit 1
