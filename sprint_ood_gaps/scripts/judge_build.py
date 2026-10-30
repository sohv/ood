"""Judge-baseline inputs (designs/judge_baseline.md). For the 90 blind-labelled topics: J-same = each shown prompt with its
screen neighbour (MiniLM nearest, as the user saw it, cosine dropped); J-50 = each shown prompt with its 5 nearest Dolci
first turns by tf-idf (exp21's vectoriser settings, fit on all 1,710,578 SFT+DPO first turns, texts cut to 2,000 chars for
fitting), neighbours cut to 300 chars. Exact copies kept, as on the screens. Writes runs/judge/inputs.jsonl."""
import json, time
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
ROOT = Path(__file__).resolve().parents[1]; OUT = ROOT / "runs/judge"; OUT.mkdir(parents=True, exist_ok=True)
IDX = ROOT / "runs/negatives/index"
items = json.load(open(ROOT / "runs/rare_gaps/labels/items.json"))
assert len(items) == 90 and all(len(it["examples"]) == 10 for it in items)
meta = pd.read_parquet(IDX / "mix_meta.parquet")
sft = pd.read_parquet(ROOT / "external/hf/detectood_ood/data/processed/exp31/dolci_sft_first_user.parquet", columns=["first_user"])
sft = sft[sft["first_user"].notna()].reset_index(drop=True)["first_user"]
from huggingface_hub import hf_hub_download
paths = [hf_hub_download("allenai/Dolci-Instruct-DPO", f"data/train-{i:05d}-of-00004.parquet", repo_type="dataset") for i in range(4)]
dpo = pd.concat([pd.read_parquet(p, columns=["chosen"]) for p in paths], ignore_index=True)["chosen"].map(lambda m: next(x["content"] for x in m if x["role"] == "user"))
mix = [sft.iat[int(r)] if p == "sft" else dpo.iat[int(r)] for p, r in zip(meta.part, meta.row)]
assert len(mix) == 1_710_578, len(mix)
t0 = time.time()
vec = TfidfVectorizer(sublinear_tf=True, min_df=2, max_features=200_000).fit([m[:2000] for m in mix])
X = vec.transform([m[:2000] for m in mix]).T.tocsr()
print(f"tf-idf {X.shape}, nnz {X.nnz}, {time.time() - t0:.0f}s", flush=True)
queries = [e["text"] for it in items for e in it["examples"]]
nb = np.zeros((len(queries), 5), dtype=np.int64); nb_s = np.zeros((len(queries), 5), dtype=np.float32)
for b in range(0, len(queries), 50):
    s = (vec.transform(queries[b:b + 50]) @ X).toarray()
    top = np.argpartition(-s, 5, axis=1)[:, :5]
    for r in range(len(top)):
        o = top[r][np.argsort(-s[r, top[r]])]; nb[b + r], nb_s[b + r] = o, s[r, o]
with open(OUT / "inputs.jsonl", "w") as f:
    for k, it in enumerate(items):
        rows = []
        for e_i, e in enumerate(it["examples"]):
            q = k * 10 + e_i
            rows.append({"text": e["text"], "screen_nearest": e["nearest"], "screen_sim": e["sim"],
                         "tfidf_mix_idx": nb[q].tolist(), "tfidf_sim": nb_s[q].round(4).tolist(), "tfidf_nearest": [mix[j][:300] for j in nb[q]]})
        f.write(json.dumps({"id": it["id"], "examples": rows}, ensure_ascii=False) + "\n")
print("wrote", OUT / "inputs.jsonl", f"{time.time() - t0:.0f}s; median top-1 tf-idf sim {np.median(nb_s[:, 0]):.3f}")
