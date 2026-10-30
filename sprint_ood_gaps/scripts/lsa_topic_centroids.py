"""Per-topic centroids in the tf-idf SVD space of scripts/topic_split_partition_lsa.py (same settings), for grouping
the tf-idf partition's topics without MiniLM (designs/one_sided_negatives.md step 8). Writes runs/topic_split/lsa_topic_centroids.npz."""
import json, time
from pathlib import Path
import numpy as np, faiss
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
ROOT = Path(__file__).resolve().parents[1]; IDX = ROOT / "runs/negatives/index"; OUT = ROOT / "runs/topic_split"
def log(*a): print(time.strftime("%H:%M:%S"), *a, flush=True)
texts = [json.loads(l)["text"][:4000] for l in open(IDX / "traffic_rows.jsonl")]
X = TfidfVectorizer(min_df=5, max_features=300_000, sublinear_tf=True, dtype=np.float32).fit_transform(texts); log("tfidf", X.shape)
Z = TruncatedSVD(256, algorithm="randomized", n_iter=4, random_state=0).fit_transform(X).astype(np.float32); faiss.normalize_L2(Z); log("svd")
lab = np.load(OUT / "lsa_labels.npy"); topics = np.unique(lab)
C = np.stack([Z[lab == t].mean(0) for t in topics]); faiss.normalize_L2(C)
np.savez(OUT / "lsa_topic_centroids.npz", topics=topics, centroids=C); log("done", C.shape)
