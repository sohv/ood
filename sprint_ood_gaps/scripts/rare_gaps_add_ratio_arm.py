"""Add a ratio-only arm to the blind labelling set (decided 2026-10-23 22:55 UTC after the register test and two outside
reviews; one label existed). Arm C: 30 topics drawn at random (seed 1) from Sohan's pipeline_million top 293 that are NOT in the
distance top 293 (same mean-top-10 ranking the distance arm used). Same item format. The 30 are inserted at random
positions among the unlabelled items; already-labelled item order is kept. key.json gains the arm."""
import json, hashlib
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[1]; L = ROOT / "runs/rare_gaps/labels"
IDX = ROOT / "runs/negatives/index"; D = ROOT / "external/hf/detectood_ood/results/raw/261021_exp37_gap_pipeline_v2"
items = json.load(open(L / "items.json")); key = json.load(open(L / "key.json"))
labelled = {"ta97306b8"}
tp = pd.read_csv(ROOT / "runs/rare_gaps/topics.csv")
pool = tp[(tp.his_rank <= 293) & (tp.rare_rank > 293)]
rng = np.random.default_rng(1)
arm = pool.iloc[rng.choice(len(pool), 30, replace=False)]
lab = np.load(D / "labels.npy"); z = np.load(ROOT / "runs/rare_gaps/one_sided.npz"); top_s, top_i = z["top_s"], z["top_i"]
text = [json.loads(l)["text"] for l in open(IDX / "traffic_rows.jsonl")]
meta = pd.read_parquet(IDX / "mix_meta.parquet")
sft = pd.read_parquet(ROOT / "external/hf/detectood_ood/data/processed/exp31/dolci_sft_first_user.parquet", columns=["first_user"])
sft = sft[sft["first_user"].notna()].reset_index(drop=True)["first_user"]
from huggingface_hub import hf_hub_download
paths = [hf_hub_download("allenai/Dolci-Instruct-DPO", f"data/train-{i:05d}-of-00004.parquet", repo_type="dataset") for i in range(4)]
dpo = pd.concat([pd.read_parquet(p, columns=["chosen"]) for p in paths], ignore_index=True)["chosen"].map(lambda m: next(x["content"] for x in m if x["role"] == "user"))
def mix_text(j): return sft.iat[int(meta.row.iat[j])] if meta.part.iat[j] == "sft" else dpo.iat[int(meta.row.iat[j])]
new = []
for _, r in arm.iterrows():
    mem = np.where(lab == int(r.lab))[0]; pick = rng.choice(mem, min(10, len(mem)), replace=False)
    iid = "t" + hashlib.sha1(f"rare-{int(r.lab)}".encode()).hexdigest()[:8]
    assert iid not in key
    key[iid] = {"arm": "ratio_only", "topic": int(r.lab), "rare_rank": int(r.rare_rank), "his_rank": int(r.his_rank), "size": int(r["size"])}
    new.append({"id": iid, "size": int(r["size"]), "examples": [{"text": text[j][:600], "nearest": mix_text(int(top_i[j, 0]))[:400], "sim": round(float(top_s[j, 0]), 3)} for j in pick]})
done = [it for it in items if it["id"] in labelled]; rest = [it for it in items if it["id"] not in labelled] + new
rest = [rest[k] for k in rng.permutation(len(rest))]
items = done + rest
for k, it in enumerate(items): it["order"] = k
(L / "items.json").write_text(json.dumps(items, indent=1, ensure_ascii=False)); (L / "key.json").write_text(json.dumps(key, indent=1))
print(len(items), "items;", {a: sum(v["arm"] == a for v in key.values()) for a in ("rare_only", "ratio_only", "control")}, "| first item still", items[0]["id"])
t = open(L / "page_template.html").read().replace("All 60 labelled", "All labelled")
s = json.dumps(items, ensure_ascii=False).replace("</", "<\\/")
open(L / "rare_gap_labels.html", "w").write(t.replace("__ITEMS__", s)); open(L / "page_template.html", "w").write(t)
