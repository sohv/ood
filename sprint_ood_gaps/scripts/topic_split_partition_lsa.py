"""Second topic partition for designs/one_sided_negatives.md step 2: k-means (2,000) on tf-idf -> 256-d SVD of all
992,933 traffic prompts, so detectors can be checked on a partition not built in MiniLM space.
Writes runs/topic_split/lsa_labels.npy."""
import json, time
from pathlib import Path
import numpy as np, faiss
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

ROOT = Path(__file__).resolve().parents[1]; IDX = ROOT / "runs/negatives/index"; OUT = ROOT / "runs/topic_split"; OUT.mkdir(parents=True, exist_ok=True)
def log(*a): print(time.strftime("%H:%M:%S"), *a, flush=True)
texts = [json.loads(l)["text"][:4000] for l in open(IDX / "traffic_rows.jsonl")]
log("texts", len(texts))
X = TfidfVectorizer(min_df=5, max_features=300_000, sublinear_tf=True, dtype=np.float32).fit_transform(texts); log("tfidf", X.shape)
Z = TruncatedSVD(256, algorithm="randomized", n_iter=4, random_state=0).fit_transform(X).astype(np.float32); log("svd")
faiss.normalize_L2(Z)
km = faiss.Kmeans(256, 2000, niter=25, seed=0, verbose=False, spherical=True, max_points_per_centroid=500)
km.train(Z); _, lab = km.index.search(Z, 1); lab = lab[:, 0]
np.save(OUT / "lsa_labels.npy", lab); sz = np.bincount(lab, minlength=2000)
log("done; cluster size min/median/max", sz.min(), int(np.median(sz)), sz.max(), "empty", int((sz == 0).sum()))
