"""Write the texts for the second-embedder run, in the same row order as the MiniLM index (runs/negatives/index):
traffic.txt.gz and mix.txt.gz, one JSON string per line, cut at 8,000 characters. Hash files are copied alongside."""
import gzip, json, shutil
from pathlib import Path
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]; IDX = ROOT / "runs/negatives/index"; OUT = ROOT / "runs/robust/upload"
CUT = 8000
with gzip.open(OUT / "traffic.txt.gz", "wt") as f:
    n = 0
    for l in open(IDX / "traffic_rows.jsonl"):
        f.write(json.dumps(json.loads(l)["text"][:CUT], ensure_ascii=False) + "\n"); n += 1
print("traffic", n)
meta = pd.read_parquet(IDX / "mix_meta.parquet")
sft = pd.read_parquet(ROOT / "external/hf/detectood_ood/data/processed/exp31/dolci_sft_first_user.parquet", columns=["first_user"])
sft = sft[sft["first_user"].notna()].reset_index(drop=True)["first_user"]
from huggingface_hub import hf_hub_download
paths = [hf_hub_download("allenai/Dolci-Instruct-DPO", f"data/train-{i:05d}-of-00004.parquet", repo_type="dataset") for i in range(4)]
dpo = pd.concat([pd.read_parquet(p, columns=["chosen"]) for p in paths], ignore_index=True)["chosen"].map(lambda m: next(x["content"] for x in m if x["role"] == "user"))
with gzip.open(OUT / "mix.txt.gz", "wt") as f:
    for part, row in zip(meta.part, meta.row):
        t = sft.iat[int(row)] if part == "sft" else dpo.iat[int(row)]
        f.write(json.dumps(t[:CUT], ensure_ascii=False) + "\n")
print("mix", len(meta))
for h in ("traffic_hash.npy", "mix_hash.npy"): shutil.copy(IDX / h, OUT / h)
