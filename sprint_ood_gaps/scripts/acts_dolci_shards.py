"""Dolci first turns in mix order (runs/negatives/index/mix_meta.parquet) as 6 jsonl shards {idx, text} for layer-20 extraction
(designs/one_sided_negatives.md step 9). Rows assigned by a seeded permutation so shards have similar length mixes."""
import json
from pathlib import Path
import numpy as np, pandas as pd
from huggingface_hub import hf_hub_download
ROOT = Path(__file__).resolve().parents[1]; IDX = ROOT / "runs/negatives/index"; OUT = ROOT / "runs/acts_dolci"
meta = pd.read_parquet(IDX / "mix_meta.parquet")
sft = pd.read_parquet(ROOT / "external/hf/detectood_ood/data/processed/exp31/dolci_sft_first_user.parquet", columns=["first_user"])
sft = sft[sft["first_user"].notna()].reset_index(drop=True)["first_user"]
dpo = pd.concat([pd.read_parquet(hf_hub_download("allenai/Dolci-Instruct-DPO", f"data/train-{i:05d}-of-00004.parquet", repo_type="dataset"),
                                 columns=["chosen"]) for i in range(4)], ignore_index=True)["chosen"]
text = [sft.iat[int(r)] if p == "sft" else next(x["content"] for x in dpo.iat[int(r)] if x["role"] == "user") for p, r in zip(meta.part, meta.row)]
perm = np.random.default_rng(0).permutation(len(text))
for s in range(6):
    with open(OUT / f"shard{s}.jsonl", "w") as f:
        for i in sorted(perm[s::6].tolist()): f.write(json.dumps({"idx": int(i), "text": text[i]}) + "\n")
print(len(text), "rows")
