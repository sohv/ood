"""Add J-hyb neighbours to runs/judge/inputs.jsonl (designs/judge_baseline.md amendment): per shown prompt, MiniLM top 5 from
runs/rare_gaps/one_sided.npz plus the tf-idf top 5 already stored, duplicates removed (MiniLM first), cut to 300 chars.
Shown prompts are matched to traffic rows by their first 600 chars; where copies share that prefix, the first row whose
top neighbour equals the screen's (text and cosine) is used, asserted."""
import json
from collections import defaultdict
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[1]; OUT = ROOT / "runs/judge"; IDX = ROOT / "runs/negatives/index"
z = np.load(ROOT / "runs/rare_gaps/one_sided.npz"); top_s, top_i = z["top_s"], z["top_i"]
rows = defaultdict(list)
for j, l in enumerate(open(IDX / "traffic_rows.jsonl")): rows[json.loads(l)["text"][:600]].append(j)
meta = pd.read_parquet(IDX / "mix_meta.parquet")
sft = pd.read_parquet(ROOT / "external/hf/detectood_ood/data/processed/exp31/dolci_sft_first_user.parquet", columns=["first_user"])
sft = sft[sft["first_user"].notna()].reset_index(drop=True)["first_user"]
from huggingface_hub import hf_hub_download
paths = [hf_hub_download("allenai/Dolci-Instruct-DPO", f"data/train-{i:05d}-of-00004.parquet", repo_type="dataset") for i in range(4)]
dpo = pd.concat([pd.read_parquet(p, columns=["chosen"]) for p in paths], ignore_index=True)["chosen"].map(lambda m: next(x["content"] for x in m if x["role"] == "user"))
def mix_text(j): return sft.iat[int(meta.row.iat[j])] if meta.part.iat[j] == "sft" else dpo.iat[int(meta.row.iat[j])]
topics = [json.loads(l) for l in open(OUT / "inputs.jsonl")]; n_amb = 0
for t in topics:
    for e in t["examples"]:
        cand = [j for j in rows[e["text"]] if mix_text(int(top_i[j, 0]))[:400] == e["screen_nearest"] and round(float(top_s[j, 0]), 3) == e["screen_sim"]]
        assert cand, e["text"][:80]; n_amb += len(rows[e["text"]]) > 1; j = cand[0]
        mini = [int(k) for k in top_i[j, :5]]
        hyb = mini + [k for k in e["tfidf_mix_idx"] if k not in mini]
        e.update(traffic_row=j, minilm_mix_idx=mini, minilm_sim=top_s[j, :5].round(4).tolist(), hybrid_mix_idx=hyb, hybrid_nearest=[mix_text(k)[:300] for k in hyb])
with open(OUT / "inputs.jsonl", "w") as f:
    for t in topics: f.write(json.dumps(t, ensure_ascii=False) + "\n")
n = [len(e["hybrid_mix_idx"]) for t in topics for e in t["examples"]]
print(f"{n_amb} shown prompts had shared-prefix copies; hybrid neighbours per prompt: mean {np.mean(n):.2f}, min {min(n)}, max {max(n)}")
