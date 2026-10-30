"""Score the topic split test (designs/one_sided_negatives.md step 2) with Olmo-3-7B-Instruct-DPO activations.
Inputs: runs/pod_sync/acts-traffic-*/acts_*.npz (Sohan's extractor: content-token mean at layers 16/20/24).
kNN as Sohan's src/metrics/detectors.py KNN: divide by the reference's per-dimension std, unit-normalise, k-th neighbour
distance sqrt(2 - 2 cos). Fit on the V1 reference only. Adds act_* columns to each partition's topics table.
Usage: topic_split_acts.py minilm lsa minilm_grouped lsa_grouped"""
import glob, json, sys, time
from pathlib import Path
import numpy as np, pandas as pd, torch, faiss
faiss.omp_set_num_threads(8)

ROOT = Path(__file__).resolve().parents[1]; ACT = ROOT / "runs/acts_traffic"; KS = [1, 10, 50]; LAYERS = [16, 20, 24]
def log(*a): print(time.strftime("%H:%M:%S"), *a, flush=True)

# assemble one memmap per layer, rows in traffic order
N = 992933
if not (ACT / "layer20.npy").exists():
    mm = {l: np.lib.format.open_memmap(ACT / f"layer{l}.npy", "w+", np.float16, (N, 4096)) for l in LAYERS}
    seen = np.zeros(N, bool); ntok = np.zeros(N, np.int32)
    for f in sorted(glob.glob(str(ROOT / "runs/pod_sync/acts-traffic-*/acts_*.npz"))):
        z = np.load(f); idx = z["idx"]; seen[idx] = True; ntok[idx] = z["n_content_tokens"]
        for j, l in enumerate(LAYERS): mm[l][idx] = z["cmean"][:, j]
    for m in mm.values(): m.flush()
    np.save(ACT / "seen.npy", seen); np.save(ACT / "n_content_tokens.npy", ntok)
    log(f"assembled {seen.sum()} of {N} rows")
seen = np.load(ACT / "seen.npy"); assert seen.all(), f"missing activations for {(~seen).sum()} rows"

for part in sys.argv[1:]:
    D = ROOT / "runs/topic_split" / part; sp = json.load(open(D / "split.json"))
    split = pd.read_parquet(D / "split.parquet"); qs = pd.read_parquet(D / "queries.parquet"); tp = pd.read_parquet(D / "topics.parquet")
    ids = [int(t) for t, v in sp["ood"].items() if v == 0]
    ref = split[(split.half == "A") & split.topic.isin(ids)].traffic_idx.values; q = qs.traffic_idx.values
    for l in LAYERS:
        X = np.load(ACT / f"layer{l}.npy", mmap_mode="r")
        R = np.asarray(X[ref], dtype=np.float32); scale = R.std(0) + 1e-6
        R /= scale; faiss.normalize_L2(R); Q = np.asarray(X[q], dtype=np.float32) / scale; faiss.normalize_L2(Q)
        ix = faiss.IndexFlatIP(R.shape[1]); ix.add(R); top, _ = ix.search(Q, max(KS)); del ix
        top = torch.from_numpy(top)
        for k in KS: qs[f"act_l{l}_knn_{k}"] = torch.sqrt(torch.clamp(2 - 2 * top[:, k - 1], min=0)).numpy()
        log(part, "layer", l, "done"); del R, Q, top
    cols = [c for c in qs.columns if c.startswith("act_")]
    tp = tp.drop(columns=[c for c in tp.columns if c.startswith("act_")]).merge(qs.groupby("topic")[cols].median().reset_index(), on="topic")
    tp.to_parquet(D / "topics.parquet"); qs.to_parquet(D / "queries.parquet"); log(part, "saved")
