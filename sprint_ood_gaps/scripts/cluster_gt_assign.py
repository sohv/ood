"""Cluster ground truth step 1 (designs/cluster_gt.md): assign every Dolci first turn to its nearest pipeline_million traffic-topic centroid,
MiniLM and Qwen3-Embedding-0.6B. Writes runs/cluster_gt/assign_{minilm,qwen}.npy (topic per Dolci row, mix order) and
topic_counts.csv (traffic and Dolci members per topic, both embedders)."""
import glob
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[1]; IDX = ROOT / "runs/negatives/index"; OUT = ROOT / "runs/cluster_gt"
lab = np.load(ROOT / "external/hf/detectood_ood/results/raw/261021_exp37_gap_pipeline_v2/labels.npy"); K = lab.max() + 1
def centroids(load_traffic_chunks):
    C = np.zeros((K, 0), np.float64); off = 0
    for x in load_traffic_chunks():
        x = x.astype(np.float32); x /= np.linalg.norm(x, axis=1, keepdims=True)
        if C.shape[1] == 0: C = np.zeros((K, x.shape[1]), np.float64)
        np.add.at(C, lab[off:off + len(x)], x); off += len(x)
    assert off == len(lab); C /= np.linalg.norm(C, axis=1, keepdims=True); return C.astype(np.float32)
def assign(C, load_mix_chunks):
    out = []
    for x in load_mix_chunks():
        x = x.astype(np.float32)
        for s in range(0, len(x), 50000): out.append((x[s:s + 50000] @ C.T).argmax(1).astype(np.int16))
    return np.concatenate(out)
t_mini = np.load(IDX / "traffic.npy", mmap_mode="r"); m_mini = np.load(IDX / "mix.npy", mmap_mode="r")
def chunks(a, n=200000):
    def f():
        for s in range(0, len(a), n): yield np.asarray(a[s:s + n])
    return f
shards = lambda pat: (lambda: (np.load(p) for p in sorted(glob.glob(str(ROOT / f"runs/robust/shards/{pat}_*.npy")))))
res = {}
for name, tl, ml in (("minilm", chunks(t_mini), chunks(m_mini)), ("qwen", shards("traffic"), shards("mix"))):
    C = centroids(tl); a = assign(C, ml); np.save(OUT / f"assign_{name}.npy", a); res[name] = np.bincount(a, minlength=K); print(name, "assigned", len(a), flush=True)
pd.DataFrame({"topic": np.arange(K), "traffic_n": np.bincount(lab, minlength=K), "dolci_minilm": res["minilm"], "dolci_qwen": res["qwen"]}).to_csv(OUT / "topic_counts.csv", index=False)
