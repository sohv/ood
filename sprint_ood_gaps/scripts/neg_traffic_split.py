"""Reproduce neg_index.build_traffic's row order without embedding, so the tail of the new-LMSYS texts can be embedded
elsewhere (a GPU) and stitched back. Writes runs/negatives/index/traffic_rows.jsonl, traffic_hash.npy,
split.json ({n_old, n_new}) and gpu_texts_from_<START>.jsonl.gz (new texts [START:], cut to 1,000 chars, in order)."""
import gzip, json, sys, time
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from neg_index import ROOT, HF, OUT, EXP09, EXP31, NEW_LMSYS_SHARDS, EMBED_CHARS, normalise, text_hash, load_jsonl, log

START = int(sys.argv[1]) if len(sys.argv) > 1 else 0
c9 = json.loads((EXP09 / "config.json").read_text())
CLONE = ROOT.parent
pool9 = [{"text": r["text"], "src": "deploy"} for r in load_jsonl(CLONE / c9["wildchat_pool"])]
pool9 += [{"text": r["text"], "src": "lmsys"} for r in load_jsonl(CLONE / c9["lmsys_pool"])]
ext = [{"text": r["text"], "src": r["source"]} for r in load_jsonl(EXP31 / "traffic_ext.jsonl")]
from huggingface_hub import HfApi, hf_hub_download
files = sorted(f for f in HfApi().list_repo_files("lmsys/lmsys-chat-1m", repo_type="dataset") if f.endswith(".parquet"))
paths = [hf_hub_download("lmsys/lmsys-chat-1m", files[i], repo_type="dataset") for i in NEW_LMSYS_SHARDS]
df = pd.concat([pd.read_parquet(p, columns=["conversation", "language"]) for p in paths], ignore_index=True)
df["text"] = df["conversation"].map(lambda c: next((m["content"] for m in c if m["role"] == "user"), "")).str.strip()
df = df[df["text"].str.len() > 0]
df = df[~df["text"].map(normalise).duplicated()]
new = [{"text": t, "src": "lmsys"} for t in df["text"]]
old = pool9 + ext
seen, rows, keep_old, keep_new = set(), [], [], []
for part, store in ((old, keep_old), (new, keep_new)):
    for i, r in enumerate(part):
        nm = normalise(r["text"])
        if nm and nm not in seen:
            seen.add(nm); rows.append({**r, "norm": nm}); store.append(i)
log(f"rows {len(rows)} (old {len(keep_old)}, new {len(keep_new)})")
assert len(rows) == 992933
np.save(OUT / "traffic_hash.npy", np.array([text_hash(r["norm"]) for r in rows], dtype=np.int64))
with open(OUT / "traffic_rows.jsonl", "w") as f:
    for r in rows: f.write(json.dumps({"text": r["text"], "src": r["src"]}, ensure_ascii=False) + "\n")
json.dump({"n_old": len(keep_old), "n_new": len(keep_new), "start": START}, open(OUT / "split.json", "w"))
texts = [new[i]["text"][:EMBED_CHARS] for i in keep_new][START:]
with gzip.open(OUT / f"gpu_texts_from_{START}.jsonl.gz", "wt") as f:
    for t in texts: f.write(json.dumps({"text": t}, ensure_ascii=False) + "\n")
log(f"wrote {len(texts)} texts from {START} for the gpu")
