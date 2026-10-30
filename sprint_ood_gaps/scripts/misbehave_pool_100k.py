"""Uniform random sample of 80,000 new traffic prompts for designs/misbehave_100k.md (seed 2), excluding behaviour_distance and the 20k run's prompts.
Output: runs/misbehave100k/prompts.jsonl in Sohan's run_rollouts format, with topic, scores, source and Dolci-member flag."""
import json
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[1]; OUT = ROOT / "runs/misbehave100k"; OUT.mkdir(parents=True, exist_ok=True)
D = ROOT / "external/hf/detectood_ood/results/raw/261021_exp37_gap_pipeline_v2"
SRC = {"deploy": "wildchat", "lmsys": "lmsys", "deploy_ext": "wildchat", "lmsys_ext": "lmsys"}
rows = [json.loads(l) for l in open(ROOT / "runs/negatives/index/traffic_rows.jsonl")]
labels = np.load(D / "labels.npy"); ratio = np.load(D / "ratio.npy")
z = np.load(ROOT / "runs/rare_gaps/one_sided.npz"); dist = 1 - z["top_s"].mean(1)
th = np.load(ROOT / "runs/negatives/index/traffic_hash.npy"); member = np.isin(th, np.load(ROOT / "runs/negatives/index/mix_hash.npy"))
used = {json.loads(l)["row"] for l in open(ROOT / "runs/behaviour_distance/prompts.jsonl")} | {json.loads(l)["row"] for l in open(ROOT / "runs/misbehave/prompts.jsonl")}
ok = np.array([len(r["text"]) <= 4000 and i not in used for i, r in enumerate(rows)])
pick = np.random.default_rng(2).choice(np.where(ok)[0], 80000, replace=False)
with open(OUT / "prompts.jsonl", "w") as f:
    for i in sorted(pick):
        f.write(json.dumps({"id": f"mb_{i}", "text": rows[i]["text"], "row": int(i), "source": SRC[rows[i]["src"]], "topic": int(labels[i]),
                            "distance": round(float(dist[i]), 4), "ratio": round(float(ratio[i]), 4), "dolci_member": bool(member[i])}, ensure_ascii=True) + "\n")
print(len(pick), "prompts; member share", round(float(member[pick].mean()), 3), "; NAME_k share", round(float(np.mean(["NAME_" in rows[i]["text"] for i in pick])), 3))
