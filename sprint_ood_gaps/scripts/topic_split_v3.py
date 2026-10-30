"""Step 8 of designs/one_sided_negatives.md: grouped topic split with groups and dev/test assigned per group, 3 seeds.
Reuses runs/topic_split/<part>/ split.parquet (template-group A/B halves) and queries.parquet (25 B-half prompts per topic).
Usage: topic_split_v3.py {lsa|minilm} SEED   -> runs/topic_split/v3_<part>_s<SEED>/{assign.json, topics.parquet, queries.parquet}"""
import json, sys, time
from pathlib import Path
import numpy as np, pandas as pd, faiss
from sklearn.feature_extraction.text import TfidfVectorizer

PART, SEED = sys.argv[1], int(sys.argv[2]); ROOT = Path(__file__).resolve().parents[1]; IDX = ROOT / "runs/negatives/index"
SRC = ROOT / "runs/topic_split" / PART; OUT = ROOT / "runs/topic_split" / f"v3_{PART}_s{SEED}"; OUT.mkdir(exist_ok=True)
sys.path.insert(0, str(ROOT.parent)); from src.metrics.detectors import Mahalanobis
N_GROUPS, LAYER, SHRINK = 200, 20, 0.9
def log(*a): print(time.strftime("%H:%M:%S"), PART, SEED, *a, flush=True)

split = pd.read_parquet(SRC / "split.parquet"); qs0 = pd.read_parquet(SRC / "queries.parquet")
kept = np.array(json.load(open(SRC / "split.json"))["kept_topics"])
# groups: k-means (seed 0, fixed across assignment seeds) on topic centroids in the partition's own space
if PART == "lsa":
    z = np.load(ROOT / "runs/topic_split/lsa_topic_centroids.npz"); pos = {int(t): i for i, t in enumerate(z["topics"])}
    C = np.ascontiguousarray(z["centroids"][[pos[int(t)] for t in kept]])
else:
    X = np.load(IDX / "traffic.npy", mmap_mode="r"); lab = split.topic.values
    C = np.stack([np.asarray(X[np.flatnonzero(lab == t)[:2000]], dtype=np.float32).mean(0) for t in kept]); faiss.normalize_L2(C)
km = faiss.Kmeans(C.shape[1], N_GROUPS, niter=50, seed=0, spherical=True); km.train(C); _, g = km.index.search(C, 1)
group = dict(zip(kept.tolist(), g[:, 0].tolist()))
rng = np.random.default_rng(SEED); gids = np.unique(g)
ood_g = set(rng.choice(gids, len(gids) // 2, replace=False).tolist()); dev_g = set(rng.choice(gids, len(gids) // 2, replace=False).tolist())
arm = {t: int(group[t] in ood_g) for t in kept.tolist()}; dev = {t: group[t] in dev_g for t in kept.tolist()}
json.dump({"group": group, "ood": arm, "dev": dev}, open(OUT / "assign.json", "w"))
ref = split[(split.half == "A") & split.topic.isin([t for t in kept.tolist() if arm[t] == 0])].traffic_idx.values
q = qs0.traffic_idx.values; texts = [json.loads(l)["text"] for l in open(IDX / "traffic_rows.jsonl")]
rows = pd.DataFrame({"traffic_idx": q, "topic": qs0.topic.values, "length": qs0.length.values})
rows["ood"] = rows.topic.map(arm); rows["dev"] = rows.topic.map(dev); rows["group"] = rows.topic.map(group)
log(f"groups {len(gids)}, OOD topics {sum(arm.values())}, dev topics {sum(dev.values())}, reference {len(ref)}, queries {len(q)}")

faiss.omp_set_num_threads(6)
T = np.load(IDX / "traffic.npy", mmap_mode="r"); R = np.asarray(T[ref], dtype=np.float32); Q = np.asarray(T[q], dtype=np.float32)
faiss.normalize_L2(R); faiss.normalize_L2(Q)
ix = faiss.IndexFlatIP(R.shape[1]); ix.add(R); s, _ = ix.search(Q, 1); rows["minilm_knn1"] = 1 - s[:, 0]; del ix
mm = Mahalanobis(R[np.random.default_rng(0).choice(len(R), min(200_000, len(R)), replace=False)], shrinkage=SHRINK); rows["minilm_maha_s09"] = mm(Q); del R, Q
log("minilm done")
vec = TfidfVectorizer(min_df=2, max_features=300_000, sublinear_tf=True, dtype=np.float32)
Rt = vec.fit_transform([texts[i][:4000] for i in ref]).T.tocsr(); Qt = vec.transform([texts[i][:4000] for i in q])
rows["tfidf_nn"] = 1 - np.concatenate([(Qt[b:b + 500] @ Rt).max(1).toarray().ravel() for b in range(0, len(q), 500)]); del Rt
log("tfidf done")
ll = np.log1p([len(texts[i]) for i in ref]); rows["length_absz"] = np.abs((np.log1p(rows.length) - ll.mean()) / ll.std())
A = np.load(ROOT / f"runs/acts_traffic/layer{LAYER}.npy", mmap_mode="r")
Ra = np.asarray(A[ref], dtype=np.float32); Qa = np.asarray(A[q], dtype=np.float32)
sub = np.random.default_rng(0).choice(len(Ra), min(100_000, len(Ra)), replace=False)
ma = Mahalanobis(Ra[sub], pca_dims=256, shrinkage=SHRINK); rows[f"act_l{LAYER}_maha_pca256_s09"] = ma(Qa)
scale = Ra.std(0) + 1e-6; Ra /= scale; Qa /= scale; faiss.normalize_L2(Ra); faiss.normalize_L2(Qa)
ix = faiss.IndexFlatIP(Ra.shape[1]); ix.add(Ra); s, _ = ix.search(Qa, 1); rows[f"act_l{LAYER}_knn1"] = np.sqrt(np.clip(2 - 2 * s[:, 0], 0, None)); del ix, Ra, Qa
rows["random"] = np.random.default_rng(SEED).random(len(rows))
log("activations done")
rows.to_parquet(OUT / "queries.parquet")
dets = [c for c in rows.columns if c not in ("traffic_idx", "topic", "length", "ood", "dev", "group")]
tp = rows.groupby("topic").agg(ood=("ood", "first"), dev=("dev", "first"), group=("group", "first"), **{d: (d, "median") for d in dets}).reset_index()
tp.to_parquet(OUT / "topics.parquet"); log("saved")
