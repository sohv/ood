"""Are 'OOD' topics that score as covered just next to an ID topic? For each OOD topic: max cosine between its B-query
centroid and any ID topic's A-half centroid (MiniLM). Usage: topic_split_neighbours.py <partition>"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd, faiss
from sklearn.metrics import roc_auc_score
ROOT = Path(__file__).resolve().parents[1]; D = ROOT / "runs/topic_split" / sys.argv[1]
X = np.load(ROOT / "runs/negatives/index/traffic.npy").astype(np.float32); faiss.normalize_L2(X)
sp = pd.read_parquet(D / "split.parquet"); qs = pd.read_parquet(D / "queries.parquet"); tp = pd.read_parquet(D / "topics.parquet")
ood = json.load(open(D / "split.json"))["ood"]; ids = [int(t) for t, v in ood.items() if v == 0]
a = sp[(sp.half == "A") & sp.topic.isin(ids)]
C = np.stack([X[g.traffic_idx.values].mean(0) for _, g in a.groupby("topic")]); faiss.normalize_L2(C)
Bc = np.stack([X[g.traffic_idx.values].mean(0) for _, g in qs.groupby("topic")]); faiss.normalize_L2(Bc)
tp = tp.set_index("topic").loc[sorted(qs.topic.unique())]
tp["nearest_id_centroid"] = (Bc @ C.T).max(1)
o = tp[tp.ood == 1].copy(); o["bin"] = pd.qcut(o.nearest_id_centroid, 4, labels=["q1 far", "q2", "q3", "q4 near"])
print(o.groupby("bin", observed=True).agg(n=("ood", "size"), nearest=("nearest_id_centroid", "median"), knn1=("v1_knn_kth_1", "median")).round(3))
idm = tp[tp.ood == 0].v1_knn_kth_1.median(); print("ID topics median knn1:", round(idm, 3))
for b, g in o.groupby("bin", observed=True):
    sub = pd.concat([g, tp[tp.ood == 0]]); print(b, "AUROC vs all ID:", round(roc_auc_score(sub.ood, sub.v1_knn_kth_1), 3))
tp.reset_index().to_parquet(D / "topics_neighbours.parquet")
