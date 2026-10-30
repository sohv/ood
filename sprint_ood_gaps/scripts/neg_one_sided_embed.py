"""Embed the neg_* and pos_* sets with the index's MiniLM (paraphrase-multilingual-MiniLM-L12-v2 rev e8f8c211, text[:1000], model default length), as scripts/neg_score.py for
designs/one_sided_negatives.md step 4. Writes runs/negatives_one_sided/sets_emb.parquet (set, kind, family, i, text, hash, member, emb)."""
import glob, json, sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "scripts"))
from neg_score import normalise, text_hash
from sentence_transformers import SentenceTransformer
rows = [json.loads(l) for f in sorted(glob.glob(str(ROOT / "runs/negatives/sets/neg_*.jsonl")) + glob.glob(str(ROOT / "runs/negatives/sets/pos_*.jsonl"))) for l in open(f)]
df = pd.DataFrame(rows)[["set", "kind", "family", "i", "text"]]
df["hash"] = [text_hash(normalise(t)) for t in df.text]
df["member"] = np.isin(df.hash.values, np.load(ROOT / "runs/negatives/index/mix_hash.npy"))
m = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", device="cpu", revision="e8f8c211226b894fcb81acc59f3b34ba3efd5f42")
df["emb"] = list(m.encode([t[:1000] for t in df.text], batch_size=256, normalize_embeddings=True, show_progress_bar=False).astype(np.float32))
df.to_parquet(ROOT / "runs/negatives_one_sided/sets_emb.parquet")
print(df.groupby("set").agg(n=("text", "size"), members=("member", "sum")).to_string())
