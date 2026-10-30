"""Prompt pool for designs/behaviour_distance.md: 5 members per topic in four arms, each with a covered control matched
on source and Olmo token-length band (gap_overrefusal's rule). Output: runs/behaviour_distance/prompts.jsonl (Sohan's run_rollouts format)."""
import json
from pathlib import Path
import numpy as np, pandas as pd
from transformers import AutoTokenizer
ROOT = Path(__file__).resolve().parents[1]; OUT = ROOT / "runs/behaviour_distance"; OUT.mkdir(parents=True, exist_ok=True)
D = ROOT / "external/hf/detectood_ood/results/raw/261021_exp37_gap_pipeline_v2"
BANDS = (1, 16, 32, 64, 128, 256, 512, 1025)
SRC = {"deploy": "wildchat", "lmsys": "lmsys", "deploy_ext": "wildchat", "lmsys_ext": "lmsys"}
rng = np.random.default_rng(0)
tp = pd.read_csv(ROOT / "runs/rare_gaps/topics.csv")
key = json.load(open(ROOT / "runs/rare_gaps/labels/key.json"))
lab_ctrl = sorted(v["topic"] for v in key.values() if v["arm"] == "control")
dist = tp[(tp.rare_rank <= 293) & (tp.his_rank > 293)].lab.tolist()
gap = tp[(tp.his_rank <= 293) & (tp.rare_rank > 293)].lab.tolist()
both = tp[(tp.his_rank <= 293) & (tp.rare_rank <= 293)].lab.tolist()
neither = tp[(tp.his_rank > 293) & (tp.rare_rank > 293)]
rest = sorted(set(neither.lab) - set(lab_ctrl))
rand = lab_ctrl + sorted(rng.choice(rest, 30, replace=False).tolist())
covered = neither[(neither.median_score <= tp.median_score.median()) & (neither.median_his <= tp.median_his.median())].lab
covered = sorted(set(covered) - set(rand))
print(f"{len(dist)=} {len(gap)=} {len(both)=} {len(rand)=} {len(covered)=}")
rows = [json.loads(l) for l in open(ROOT / "runs/negatives/index/traffic_rows.jsonl")]
labels = np.load(D / "labels.npy"); assert len(labels) == len(rows)
text = [r["text"] for r in rows]; source = np.array([SRC[r["src"]] for r in rows])
short = np.array([len(t) <= 4000 for t in text])
tok = AutoTokenizer.from_pretrained("allenai/Olmo-3-7B-Instruct-DPO", revision="b33130b7de49f0c2553b5c2b3bc8409ff3e627d1")
need = np.isin(labels, dist + gap + both + rand + covered)
n_tok = np.zeros(len(rows), dtype=int); idx = np.where(need)[0]
for s in range(0, len(idx), 20000):
    b = idx[s:s + 20000]; n_tok[b] = [len(x) for x in tok([text[i] for i in b], add_special_tokens=False)["input_ids"]]
band = np.digitize(n_tok, BANDS)
members = {c: np.where((labels == c) & short)[0] for c in set(dist + gap + both + rand + covered)}
used = np.zeros(len(rows), dtype=bool); out = []
for arm, topics in (("distance_only", dist), ("gap_only", gap), ("both", both), ("random", rand)):
    for c in topics:
        for j, r in enumerate(rng.choice(members[c], 5, replace=False)):
            used[r] = True
            out.append({"id": f"{arm}_c{c}_p{j}", "text": text[r], "role": "member", "group": f"{arm}_c{c}", "arm": arm, "topic": int(c), "source": source[r], "row": int(r)})
            for tries in range(500):
                ct = covered[rng.integers(len(covered))]; m = members[ct]
                pool = m[(source[m] == source[r]) & (band[m] == band[r]) & ~used[m]]
                if len(pool): break
            assert len(pool), f"no control for {r}"
            cr = int(pool[rng.integers(len(pool))]); used[cr] = True
            out.append({"id": f"{arm}_c{c}_p{j}_c0", "text": text[cr], "role": "control", "group": f"{arm}_c{c}", "arm": arm, "topic": int(c), "source": source[cr], "row": cr, "control_topic": int(ct)})
with open(OUT / "prompts.jsonl", "w") as f:
    for o in out: f.write(json.dumps(o, ensure_ascii=True) + "\n")
print(len(out), "prompts ->", OUT / "prompts.jsonl")
