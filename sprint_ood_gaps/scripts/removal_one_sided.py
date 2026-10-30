"""Step 3 of designs/one_sided_negatives.md: removal-then-recover with a one-sided detector (MiniLM kNN distance to Dolci).
Removals: safety, multilingual, coding (step 1 family lists + exact copies) and 3 random removals of each size.
Writes runs/removal_one_sided/{queries.parquet, topics.parquet, summary.json}."""
import json, sys, time
from pathlib import Path
import numpy as np, pandas as pd, faiss
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]; IDX = ROOT / "runs/negatives/index"; OUT = ROOT / "runs/removal_one_sided"; OUT.mkdir(exist_ok=True)
sys.path.insert(0, str(ROOT / "scripts"))
def log(*a): print(time.strftime("%H:%M:%S"), *a, flush=True)
K_SEARCH, N_Q, TOP = 500, 25, 308
KINDS = {"safety": "is_harmful", "multilingual": "is_non_english", "coding": "is_coding"}
FAM = {"safety": "safety", "multilingual": "multilingual", "coding": "code"}

# step 1's family labels for mix rows
import importlib.util
src_txt = open(ROOT / "scripts/one_sided_score.py").read()
FAMILIES = eval(src_txt[src_txt.index("FAMILIES = {") + len("FAMILIES = "):src_txt.index("\n}\n") + 2])
import re
meta = pd.read_parquet(IDX / "mix_meta.parquet")
sft = pd.read_parquet(ROOT / "external/hf/detectood_ood/data/processed/exp31/dolci_sft_first_user.parquet", columns=["source_dataset", "first_user"])
sft = sft[sft["first_user"].notna()].reset_index(drop=True)
def prefix(pid):
    pid = re.sub(r"-request-.*$", "", pid); pid = re.sub(r"[-_]\d+[a-z0-9]*$", "", pid); return re.sub(r"[-_][0-9a-f]{8,}$", "", pid)
src = np.where(meta.part.values == "sft", sft["source_dataset"].reindex(meta.row.values).values, None).astype(object)
dm = meta.part.values == "dpo"; src[dm] = ["dpo:" + prefix(p) for p in meta.prompt_id.values[dm]]
mh = np.load(IDX / "mix_hash.npy"); n_mix = len(mh)

# topic labels (Sohan's grader labels, as exp80/84)
H = ROOT / "runs/removal_why/hf/results/raw"
harm = {json.loads(l)["id"]: json.loads(l)["is_harmful"] for l in open(H / "261028_exp80_safety_removal_v1/topics.jsonl")}
kl = {json.loads(l)["id"]: json.loads(l) for l in open(H / "261028_exp84_kind_labels_v1/topic_labels.jsonl")}
lab = np.load(ROOT / "external/hf/detectood_ood/results/raw/261021_exp37_gap_pipeline_v2/labels.npy")
th = np.load(IDX / "traffic_hash.npy"); member = np.isin(th, mh)
rng = np.random.default_rng(0)
q = np.sort(np.concatenate([rng.choice(c, min(N_Q, len(c)), replace=False) for t in np.unique(lab)
                            for c in [np.flatnonzero((lab == t) & ~member)] if len(c)]))
log(f"{len(q)} queries over {len(np.unique(lab[q]))} topics")

M = np.load(IDX / "mix.npy").astype(np.float32); faiss.normalize_L2(M); faiss.omp_set_num_threads(8)
T = np.load(IDX / "traffic.npy", mmap_mode="r"); Q = np.asarray(T[q], dtype=np.float32); faiss.normalize_L2(Q)
ix = faiss.IndexFlatIP(384); ix.add(M)
S, I = ix.search(Q, K_SEARCH); log("full search done")

# removals: kind families (+ exact copies) and random removals of the same size
removals = {}
for kind, fam in FAM.items():
    rows = np.isin(src, FAMILIES[fam]); rows |= np.isin(mh, mh[rows]); removals[kind] = rows
    for s in range(3):
        r = np.zeros(n_mix, bool); r[np.random.default_rng(100 + s).choice(n_mix, rows.sum(), replace=False)] = True; removals[f"random_{kind}_{s}"] = r
rows_out = pd.DataFrame({"traffic_idx": q, "topic": lab[q]})
for name, sims in [("full", S)]:
    rows_out["full_k1"] = 1 - sims[:, 0]; rows_out["full_mean5"] = 1 - sims[:, :5].mean(1)
for name, rem in removals.items():
    keep = ~rem[I]; k1 = np.full(len(q), np.nan, np.float32); m5 = np.full(len(q), np.nan, np.float32)
    ok = keep.sum(1) >= 5
    for i in np.flatnonzero(ok):
        s = S[i][keep[i]]; k1[i] = 1 - s[0]; m5[i] = 1 - s[:5].mean()
    redo = np.flatnonzero(~ok)
    if len(redo):  # all 500 nearest removed: search the reduced index
        ix2 = faiss.IndexFlatIP(384); ix2.add(M[~rem]); s2, _ = ix2.search(Q[redo], 5); del ix2
        k1[redo] = 1 - s2[:, 0]; m5[redo] = 1 - s2.mean(1)
    rows_out[f"{name}_k1"] = k1; rows_out[f"{name}_mean5"] = m5; log(name, f"removed {rem.sum()}, re-searched {len(redo)}")
rows_out.to_parquet(OUT / "queries.parquet")
# summary: scripts/removal_one_sided_summary.py
