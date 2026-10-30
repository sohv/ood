"""Step 2 of designs/one_sided_negatives.md: topic-level split test on traffic, one-sided detectors.
Usage: topic_split_test.py {minilm|lsa}
  V1: reference = A halves of ID topics only. Score B-half queries of every kept topic. Label OOD = 1.
  V2 (minilm partition only): reference = Dolci, then Dolci + ID topics' A halves; MiniLM kNN only.
Writes runs/topic_split/<partition>/{split.parquet, queries.parquet, topics.parquet}."""
import json, sys, time, hashlib
from pathlib import Path
import numpy as np, pandas as pd, faiss
from sklearn.covariance import LedoitWolf
from sklearn.feature_extraction.text import TfidfVectorizer

PART = sys.argv[1]; GROUPED = len(sys.argv) > 2 and sys.argv[2] == "grouped"; ROOT = Path(__file__).resolve().parents[1]; IDX = ROOT / "runs/negatives/index"
OUT = ROOT / "runs/topic_split" / (PART + ("_grouped" if GROUPED else "")); OUT.mkdir(parents=True, exist_ok=True)
N_Q, MIN_HALF, KS, SEED = 25, 20, [1, 5, 20], 0
def log(*a): print(time.strftime("%H:%M:%S"), *a, flush=True)

lab = (np.load(ROOT / "external/hf/detectood_ood/results/raw/261021_exp37_gap_pipeline_v2/labels.npy") if PART == "minilm"
       else np.load(ROOT / "runs/topic_split/lsa_labels.npy"))
texts = [json.loads(l)["text"] for l in open(IDX / "traffic_rows.jsonl")]; n = len(texts); assert len(lab) == n
norm = [" ".join(t.lower().split()) for t in texts]
th = np.load(IDX / "traffic_hash.npy"); member = np.isin(th, np.load(IDX / "mix_hash.npy"))
log(f"{n} traffic prompts, {member.mean():.3f} verbatim in Dolci")

# split each topic by template group (first 60 normalised characters); groups shuffled, assigned greedily to the smaller half
rng = np.random.default_rng(SEED)
key = pd.Series([s[:60] for s in norm]).astype("category").cat.codes.values
half = np.full(n, "", dtype=object); kept = []
for t in np.unique(lab):
    idx = np.flatnonzero(lab == t); g = pd.Series(idx).groupby(key[idx]).apply(list).tolist()
    if len(g) < 2: continue
    rng.shuffle(g); a, b = [], []
    for grp in g: (a if len(a) <= len(b) else b).extend(grp)
    if (~member[a]).sum() >= MIN_HALF and (~member[b]).sum() >= MIN_HALF:
        half[a] = "A"; half[b] = "B"; kept.append(t)
kept = np.array(kept); topic_arm = dict(zip(kept, rng.permutation(np.r_[np.ones(len(kept) // 2, int), np.zeros(len(kept) - len(kept) // 2, int)])))
if GROUPED:  # amendment after V1: assign ID/OOD to groups of neighbouring topics (k-means, 200 groups, on topic centroids in MiniLM)
    Xc = np.load(IDX / "traffic.npy", mmap_mode="r")
    C = np.stack([np.asarray(Xc[np.flatnonzero(lab == t)[:2000]], dtype=np.float32).mean(0) for t in kept]); faiss.normalize_L2(C)
    km = faiss.Kmeans(C.shape[1], 200, niter=50, seed=SEED, spherical=True); km.train(C); _, grp = km.index.search(C, 1); grp = grp[:, 0]
    garm = dict(zip(range(200), rng.permutation(np.r_[np.ones(100, int), np.zeros(100, int)])))
    topic_arm = {t: garm[int(g)] for t, g in zip(kept, grp)}
    json.dump({int(t): int(g) for t, g in zip(kept, grp)}, open(OUT / "groups.json", "w"))
topic_dev = dict(zip(kept, rng.random(len(kept)) < 0.5))
log(f"kept {len(kept)} of {len(np.unique(lab))} topics; OOD {sum(topic_arm.values())}")
# queries: up to 25 non-member B prompts per kept topic
q = []
for t in kept:
    c = np.flatnonzero((lab == t) & (half == "B") & ~member); q.extend(rng.choice(c, min(N_Q, len(c)), replace=False).tolist())
q = np.array(sorted(q)); ood = np.array([topic_arm[t] for t in lab[q]])
ref = np.flatnonzero((half == "A") & np.isin(lab, [t for t in kept if topic_arm[t] == 0]))
pd.DataFrame({"traffic_idx": np.arange(n), "topic": lab, "half": half, "member": member}).to_parquet(OUT / "split.parquet")
json.dump({"kept_topics": [int(t) for t in kept], "ood": {int(k): int(v) for k, v in topic_arm.items()}, "dev": {int(k): bool(v) for k, v in topic_dev.items()},
           "n_queries": int(len(q)), "n_ref_v1": int(len(ref))}, open(OUT / "split.json", "w"))
log(f"queries {len(q)}, V1 reference {len(ref)}")

X = np.load(IDX / "traffic.npy").astype(np.float32); faiss.normalize_L2(X); faiss.omp_set_num_threads(8)
Q = X[q]; rows = pd.DataFrame({"traffic_idx": q, "topic": lab[q], "ood": ood, "dev": [topic_dev[t] for t in lab[q]], "length": [len(texts[i]) for i in q]})

# V1 MiniLM kNN
ix = faiss.IndexFlatIP(384); ix.add(X[ref]); s_ref, _ = ix.search(Q, max(KS)); del ix
for k in KS: rows[f"v1_knn_kth_{k}"] = 1 - s_ref[:, k - 1]; rows[f"v1_knn_mean_{k}"] = 1 - s_ref[:, :k].mean(1)
# V1 Mahalanobis: global, and class-conditional with ID topics as classes (circular on the minilm partition)
R = X[ref]; mu = R.mean(0); P = LedoitWolf().fit(R[rng.choice(len(R), min(200_000, len(R)), replace=False)] - mu).precision_.astype(np.float32)
D = Q - mu; rows["v1_maha"] = np.einsum("ij,jk,ik->i", D, P, D)
rl = lab[ref]; cls = np.unique(rl); mus = np.stack([R[rl == c].mean(0) for c in cls])
res = R - mus[np.searchsorted(cls, rl)]; Pc = LedoitWolf(assume_centered=True).fit(res[rng.choice(len(res), min(200_000, len(res)), replace=False)]).precision_.astype(np.float32)
best = np.full(len(Q), np.inf, np.float32)
for b in range(0, len(cls), 100):
    M = mus[b:b + 100]; d = (Q @ Pc * Q).sum(1)[:, None] - 2 * (Q @ Pc) @ M.T + ((M @ Pc) * M).sum(1)[None]; best = np.minimum(best, d.min(1))
rows["v1_maha_cc"] = best; del R, res
# V1 one-sided tf-idf: max cosine to the reference, vectorizer fit on the reference only
vec = TfidfVectorizer(min_df=2, max_features=300_000, sublinear_tf=True, dtype=np.float32)
Rt = vec.fit_transform([texts[i][:4000] for i in ref]).T.tocsr(); Qt = vec.transform([texts[i][:4000] for i in q])
mx = np.zeros(len(q), np.float32)
for b in range(0, len(q), 500): mx[b:b + 500] = (Qt[b:b + 500] @ Rt).max(1).toarray().ravel()
rows["v1_tfidf"] = 1 - mx
an = vec.build_analyzer(); voc = vec.vocabulary_
rows["v1_unseen_words"] = [np.mean([w not in voc for w in an(texts[i][:4000])] or [0.0]) for i in q]
# V1 one-sided length: |z| of log length against the reference
ll = np.log1p([len(texts[i]) for i in ref]); rows["v1_length"] = np.abs((np.log1p(rows.length) - ll.mean()) / ll.std())
rows["random"] = rng.random(len(q))
log("V1 scored")

if PART == "minilm":  # V2: Dolci, then Dolci + ID A halves (merge of two searches)
    M = np.load(IDX / "mix.npy").astype(np.float32); faiss.normalize_L2(M)
    ix = faiss.IndexFlatIP(384); ix.add(M); s_mix, _ = ix.search(Q, max(KS)); del ix, M
    both = np.sort(np.concatenate([s_mix, s_ref], 1), 1)[:, ::-1]
    for k in KS:
        rows[f"v2_dolci_kth_{k}"] = 1 - s_mix[:, k - 1]; rows[f"v2_added_kth_{k}"] = 1 - both[:, k - 1]
        rows[f"v2_dolci_mean_{k}"] = 1 - s_mix[:, :k].mean(1); rows[f"v2_added_mean_{k}"] = 1 - both[:, :k].mean(1)
    log("V2 scored")

rows["text"] = [texts[i][:1500] for i in q]
rows.to_parquet(OUT / "queries.parquet")
score_cols = [c for c in rows.columns if c.startswith(("v1_", "v2_")) or c == "random"]
topics = rows.groupby("topic").agg(ood=("ood", "first"), dev=("dev", "first"), n=("ood", "size"), length=("length", "median"),
                                   **{c: (c, "median") for c in score_cols}).reset_index()
topics.to_parquet(OUT / "topics.parquet"); log("done", OUT)
