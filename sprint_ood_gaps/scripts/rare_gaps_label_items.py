"""Blind topic-labelling items for the rare-gap check (fixed before any labelling, 2026-10-23).
Arm A, rare-only: the first 30 topics by median distance to Dolci (runs/rare_gaps/topics.csv) that are NOT in Sohan's
pipeline_million top 293. Arm B, control: 30 topics drawn at random (seed 0) from topics in neither top 293. Topics with >= 50
members only (as his build). Each item: 10 random members (seed 0), each with its nearest Dolci prompt and cosine.
No topic names (none exist for most topics; his deep_gap_loop labels showed names, so this is stricter). Order shuffled (seed 0);
ids are opaque. Key in runs/rare_gaps/labels/key.json; items in items.json. Labels: real_gap / partly / covered,
same scale as his deep_gap_loop."""
import json, hashlib
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[1]; OUT = ROOT / "runs/rare_gaps/labels"; OUT.mkdir(parents=True, exist_ok=True)
IDX = ROOT / "runs/negatives/index"; D = ROOT / "external/hf/detectood_ood/results/raw/261021_exp37_gap_pipeline_v2"
tp = pd.read_csv(ROOT / "runs/rare_gaps/topics.csv")
rare = tp[(tp.his_rank > 293)].sort_values("rare_rank").head(30)
pool = tp[(tp.his_rank > 293) & (tp.rare_rank > 293)]
rng = np.random.default_rng(0)
ctrl = pool.iloc[rng.choice(len(pool), 30, replace=False)]
lab = np.load(D / "labels.npy"); z = np.load(ROOT / "runs/rare_gaps/one_sided.npz"); top_s, top_i = z["top_s"], z["top_i"]
text = [json.loads(l)["text"] for l in open(IDX / "traffic_rows.jsonl")]
meta = pd.read_parquet(IDX / "mix_meta.parquet")
sft = pd.read_parquet(ROOT / "external/hf/detectood_ood/data/processed/exp31/dolci_sft_first_user.parquet", columns=["first_user"])
sft = sft[sft["first_user"].notna()].reset_index(drop=True)["first_user"]
from huggingface_hub import hf_hub_download
paths = [hf_hub_download("allenai/Dolci-Instruct-DPO", f"data/train-{i:05d}-of-00004.parquet", repo_type="dataset") for i in range(4)]
dpo = pd.concat([pd.read_parquet(p, columns=["chosen"]) for p in paths], ignore_index=True)["chosen"].map(lambda m: next(x["content"] for x in m if x["role"] == "user"))
def mix_text(j): return sft.iat[int(meta.row.iat[j])] if meta.part.iat[j] == "sft" else dpo.iat[int(meta.row.iat[j])]
items, key = [], {}
for arm, df in (("rare_only", rare), ("control", ctrl)):
    for _, r in df.iterrows():
        mem = np.where(lab == int(r.lab))[0]; pick = rng.choice(mem, min(10, len(mem)), replace=False)
        iid = "t" + hashlib.sha1(f"rare-{int(r.lab)}".encode()).hexdigest()[:8]
        key[iid] = {"arm": arm, "topic": int(r.lab), "rare_rank": int(r.rare_rank), "his_rank": int(r.his_rank), "size": int(r["size"])}
        items.append({"id": iid, "size": int(r["size"]), "examples": [{"text": text[j][:600], "nearest": mix_text(int(top_i[j, 0]))[:400], "sim": round(float(top_s[j, 0]), 3)} for j in pick]})
order = rng.permutation(len(items)); items = [items[k] for k in order]
for k, it in enumerate(items): it["order"] = k
(OUT / "items.json").write_text(json.dumps(items, indent=1, ensure_ascii=False)); (OUT / "key.json").write_text(json.dumps(key, indent=1))
print(len(items), "items;", sum(v["arm"] == "rare_only" for v in key.values()), "rare-only,", sum(v["arm"] == "control" for v in key.values()), "control; size of items.json", (OUT / "items.json").stat().st_size)
