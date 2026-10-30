"""Step 7 of designs/one_sided_negatives.md: blind label screens for the corrected loop's R50 gaps vs random below-line topics.
30 of the 54 R50 gaps (lead topic) + 20 random topics ranked > 293, excluding topics labelled on 24 Oct; 10 non-member prompts
per topic, each with its 2 nearest Dolci prompts by MiniLM and 2 by tf-idf. Writes runs/loop_labels/{items.json, key.json}."""
import hashlib, json
from pathlib import Path
import numpy as np, pandas as pd, faiss
from sklearn.feature_extraction.text import TfidfVectorizer

ROOT = Path(__file__).resolve().parents[1]; IDX = ROOT / "runs/negatives/index"; OUT = ROOT / "runs/loop_labels"; OUT.mkdir(exist_ok=True)
rng = np.random.default_rng(0)
d = json.load(open(ROOT / "runs/loop_onesided/v2/decision.json"))
gap_topics = [g[0]["topic"] for g in d["gaps"]["R50"]]
old = {v["topic"] for v in json.load(open(ROOT / "runs/rare_gaps/labels/key.json")).values()}
tp = pd.read_parquet(ROOT / "runs/removal_one_sided/topics.parquet").sort_values("full_k1", ascending=False).reset_index(drop=True)
rank = {int(t): i + 1 for i, t in enumerate(tp.topic)}
loop_pick = rng.choice([t for t in gap_topics if t not in old], 30, replace=False).tolist()
below = [int(t) for t in tp.topic if rank[int(t)] > 293 and int(t) not in old]
rand_pick = rng.choice(below, 20, replace=False).tolist()
picks = [(t, "loop") for t in loop_pick] + [(t, "random") for t in rand_pick]
order = rng.permutation(len(picks))

lab = np.load(ROOT / "external/hf/detectood_ood/results/raw/261021_exp37_gap_pipeline_v2/labels.npy")
texts = [json.loads(l)["text"] for l in open(IDX / "traffic_rows.jsonl")]
member = np.isin(np.load(IDX / "traffic_hash.npy"), np.load(IDX / "mix_hash.npy"))
shown = {t: rng.choice(np.flatnonzero((lab == t) & ~member), 10, replace=False) for t, _ in picks}
qidx = np.concatenate([shown[t] for t, _ in picks])

# Dolci texts in mix order
meta = pd.read_parquet(IDX / "mix_meta.parquet")
sft = pd.read_parquet(ROOT / "external/hf/detectood_ood/data/processed/exp31/dolci_sft_first_user.parquet", columns=["first_user"])
sft = sft[sft["first_user"].notna()].reset_index(drop=True)["first_user"]
from huggingface_hub import hf_hub_download
dpo = pd.concat([pd.read_parquet(hf_hub_download("allenai/Dolci-Instruct-DPO", f"data/train-{i:05d}-of-00004.parquet", repo_type="dataset"),
                                 columns=["chosen"]) for i in range(4)], ignore_index=True)["chosen"]
mix_text = [sft.iat[int(r)] if p == "sft" else next(x["content"] for x in dpo.iat[int(r)] if x["role"] == "user") for p, r in zip(meta.part, meta.row)]

M = np.load(IDX / "mix.npy").astype(np.float32); faiss.normalize_L2(M)
Q = np.load(IDX / "traffic.npy", mmap_mode="r")[qidx].astype(np.float32); faiss.normalize_L2(Q)
ix = faiss.IndexFlatIP(384); ix.add(M); _, nn_m = ix.search(Q, 2); del ix, M
vec = TfidfVectorizer(ngram_range=(1, 2), min_df=3, sublinear_tf=True, max_features=800_000, dtype=np.float32)
X = vec.fit_transform([t[:4000] for t in mix_text]).T.tocsr()
nn_t = []
for b in range(0, len(qidx), 50):
    S = (vec.transform([texts[i][:4000] for i in qidx[b:b + 50]]) @ X).toarray(); nn_t += np.argsort(-S, 1)[:, :2].tolist()
nb = {}
for k, i in enumerate(qidx):
    seen, out = set(), []
    for j in list(nn_m[k]) + list(nn_t[k]):
        s = mix_text[j][:300]
        if s not in seen: seen.add(s); out.append(s)
    nb[int(i)] = out

items, key = [], {}
for pos, oi in enumerate(order):
    t, arm = picks[oi]
    iid = "L" + hashlib.blake2b(f"{t}-{arm}-s0".encode(), digest_size=4).hexdigest()
    items.append({"id": iid, "size": int((lab == t).sum()), "order": pos,
                  "examples": [{"text": texts[i][:1500], "verbatim": False, "nb": nb[int(i)]} for i in shown[t]]})
    key[iid] = {"topic": int(t), "arm": arm, "rank": rank[int(t)]}
json.dump(items, open(OUT / "items.json", "w"), ensure_ascii=False); json.dump(key, open(OUT / "key.json", "w"), indent=1)
print(len(items), "items;", sum(v["arm"] == "loop" for v in key.values()), "loop,", sum(v["arm"] == "random" for v in key.values()), "random")
