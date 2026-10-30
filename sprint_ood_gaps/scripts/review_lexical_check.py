"""designs/review_lexical_check.md: tf-idf retrieval over Dolci for the user's real-gap topics. Writes runs/review/lexical/."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
ROOT = Path(__file__).resolve().parents[1]; OUT = ROOT / "runs/review/lexical"; OUT.mkdir(parents=True, exist_ok=True)
IDX = ROOT / "runs/negatives/index"; D = ROOT / "external/hf/detectood_ood/results/raw/261021_exp37_gap_pipeline_v2"; L = ROOT / "runs/rare_gaps/labels"
key = json.load(open(L / "key.json")); dump = {d["id"]: d["data"]["verdict"] for d in json.load(open(L / "labels_dump.json"))}
items = {it["id"]: it for it in json.load(open(L / "items.json"))}
ids = [i for i in key if dump[i] == "real_gap"]; print("topics", len(ids), pd.Series([key[i]["arm"] for i in ids]).value_counts().to_dict())
lab = np.load(D / "labels.npy"); th = np.load(IDX / "traffic_hash.npy"); mh = np.load(IDX / "mix_hash.npy"); member = np.isin(th, mh)
text = [json.loads(l)["text"] for l in open(IDX / "traffic_rows.jsonl")]
meta = pd.read_parquet(IDX / "mix_meta.parquet")
sft = pd.read_parquet(ROOT / "external/hf/detectood_ood/data/processed/exp31/dolci_sft_first_user.parquet", columns=["first_user"])
sft = sft[sft["first_user"].notna()].reset_index(drop=True)["first_user"]
from huggingface_hub import hf_hub_download
paths = [hf_hub_download("allenai/Dolci-Instruct-DPO", f"data/train-{i:05d}-of-00004.parquet", repo_type="dataset") for i in range(4)]
dpo = pd.concat([pd.read_parquet(p, columns=["chosen"]) for p in paths], ignore_index=True)["chosen"].map(lambda m: next(x["content"] for x in m if x["role"] == "user"))
mix_text = [sft.iat[int(r)] if p == "sft" else dpo.iat[int(r)] for p, r in zip(meta.part, meta.row)]
print("mix texts", len(mix_text), flush=True)
# recover shown rows by prefix match within the topic
queries = []
for i in ids:
    mem = np.where(lab == key[i]["topic"])[0]
    for k, e in enumerate(items[i]["examples"]):
        hits = [r for r in mem if text[r].startswith(e["text"])]
        r = hits[0] if hits else None
        queries.append({"id": i, "k": k, "row": int(r) if r is not None else None, "text": text[r] if r is not None else e["text"], "minilm": e["nearest"], "sim": e["sim"], "member": bool(member[r]) if r is not None else None, "n_hits": len(hits)})
print("queries", len(queries), "unresolved", sum(q["row"] is None for q in queries), "ambiguous", sum(q["n_hits"] > 1 for q in queries), flush=True)
vec = TfidfVectorizer(ngram_range=(1, 2), min_df=3, sublinear_tf=True, max_features=800000, dtype=np.float32)
X = vec.fit_transform(mix_text); print("tfidf", X.shape, flush=True)
Q = vec.transform([q["text"] for q in queries]); S = (X @ Q.T).toarray()  # (n_mix, n_q)
for j, q in enumerate(queries):
    top = np.argsort(-S[:, j])[:3]
    q["lexical"] = [{"text": mix_text[t][:400], "score": round(float(S[t, j]), 3)} for t in top]
json.dump(queries, open(OUT / "retrieval.json", "w"), indent=1, ensure_ascii=False)
rng = np.random.default_rng(0); order = rng.permutation(len(ids))
with open(OUT / "sheets.md", "w") as f:
    for n, oi in enumerate(order):
        i = ids[oi]; f.write(f"\n\n# Topic {n + 1} (id {i})\n")
        for q in [q for q in queries if q["id"] == i]:
            f.write(f"\n## Prompt {q['k'] + 1}{'  [this exact prompt is in Dolci]' if q['member'] else ''}\n{q['text'][:800]}\n\n- MiniLM nearest (cos {q['sim']}): {q['minilm'][:400]}\n")
            for m, lx in enumerate(q["lexical"]): f.write(f"- Lexical {m + 1} (tfidf {lx['score']}): {lx['text']}\n")
json.dump({f"Topic {n + 1}": ids[oi] for n, oi in enumerate(order)}, open(OUT / "sheet_key.json", "w"), indent=1)
print("done ->", OUT)
