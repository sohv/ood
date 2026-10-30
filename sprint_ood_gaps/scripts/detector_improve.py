"""Step 5 of designs/one_sided_negatives.md: candidate one-sided detectors on the topic split test.
(b) Mahalanobis on Olmo activations in the reference's top-256 principal axes (Ledoit-Wolf), layers 16/20/24;
(c) kNN k=1 on layers 16+20+24 (each std-scaled, unit-normalised; cosine of the concatenation = mean of per-layer cosines),
    exact over the union of each layer's 200 nearest reference rows;
(a)/(d) rank-averages are built in the analysis from topic scores. Usage: detector_improve.py <partition> [b] [c]
Writes runs/topic_split/<partition>/candidates_<b|c>.parquet (topic-level medians)."""
import json, sys, time
from pathlib import Path
import numpy as np, pandas as pd, faiss
from sklearn.covariance import LedoitWolf

ROOT = Path(__file__).resolve().parents[1]; ACT = ROOT / "runs/acts_traffic"; LAYERS = [16, 20, 24]
faiss.omp_set_num_threads(6)
def log(*a): print(time.strftime("%H:%M:%S"), *a, flush=True)
part, which = sys.argv[1], set(sys.argv[2:]) or {"b", "c"}
D = ROOT / "runs/topic_split" / part; sp = json.load(open(D / "split.json"))
split = pd.read_parquet(D / "split.parquet"); qs = pd.read_parquet(D / "queries.parquet", columns=["traffic_idx", "topic"])
ids = [int(t) for t, v in sp["ood"].items() if v == 0]
ref = split[(split.half == "A") & split.topic.isin(ids)].traffic_idx.values; q = qs.traffic_idx.values

if "b" in which:
    t0 = time.time(); out = qs[["topic"]].copy()
    for l in LAYERS:
        X = np.load(ACT / f"layer{l}.npy", mmap_mode="r"); R = np.asarray(X[ref], dtype=np.float32); mu = R.mean(0)
        C = np.zeros((R.shape[1], R.shape[1]))
        for b in range(0, len(R), 20000): Z = (R[b:b + 20000] - mu).astype(np.float64); C += Z.T @ Z
        V = np.linalg.eigh(C / len(R))[1][:, ::-1][:, :256].astype(np.float32)
        Zr = (R - mu) @ V; del R
        P = LedoitWolf().fit(Zr).precision_.astype(np.float32)
        Zq = (np.asarray(X[q], dtype=np.float32) - mu) @ V; Zq -= Zr.mean(0)
        out[f"maha_pca256_l{l}"] = np.einsum("ij,jk,ik->i", Zq, P, Zq); log(part, "b layer", l, f"{time.time() - t0:.0f}s")
    out.groupby("topic").median().reset_index().to_parquet(D / "candidates_b.parquet")

if "c" in which:
    t0 = time.time(); Rn, Qn, cand = [], [], []
    for l in LAYERS:
        X = np.load(ACT / f"layer{l}.npy", mmap_mode="r"); R = np.asarray(X[ref], dtype=np.float32); scale = R.std(0) + 1e-6
        R /= scale; faiss.normalize_L2(R); Q = np.asarray(X[q], dtype=np.float32) / scale; faiss.normalize_L2(Q)
        ix = faiss.IndexFlatIP(R.shape[1]); ix.add(R); _, I = ix.search(Q, 200); del ix
        Rn.append(R.astype(np.float16)); Qn.append(Q); cand.append(I); del R
        log(part, "c layer", l, f"{time.time() - t0:.0f}s")
    U = np.concatenate(cand, 1); best = np.empty(len(q), np.float32)
    for i in range(len(q)):
        u = np.unique(U[i]); best[i] = sum((Rn[j][u].astype(np.float32) @ Qn[j][i]) for j in range(3)).max() / 3
    out = qs[["topic"]].copy(); out["knn1_concat_16_20_24"] = np.sqrt(np.clip(2 - 2 * best, 0, None))
    out.groupby("topic").median().reset_index().to_parquet(D / "candidates_c.parquet"); log(part, "c done", f"{time.time() - t0:.0f}s")
