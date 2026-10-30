"""Safety-removal detectors, step 2 (designs/safety_removal_detectors.md). For arms R0 (nothing removed), R1 (WildGuardMix),
R2 (WildJailbreak), R3 (R1 + R2 + CoCoNot), Rc (random, same row count as R3, seed 0), with DPO counterparts by prompt_id prefix:
score the pool's traffic prompts against the reduced Dolci mix and the full traffic pool, exactly as scripts/leave_source_out.py
(k = 50 ratio, float32, exact copies masked by hash; distance = 1 - nearest Dolci cosine). Output runs/safety_removal/scores.parquet."""
import json, sys, re, time
from pathlib import Path
import numpy as np, pandas as pd, faiss
ROOT = Path(__file__).resolve().parents[1]; IDX = ROOT / "runs/negatives/index"; OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "runs/safety_removal"
K, EXTRA = 50, 20; faiss.omp_set_num_threads(8)
def log(*a): print(time.strftime("%H:%M:%S"), *a, flush=True)
meta = pd.read_parquet(IDX / "mix_meta.parquet")
sft_src = pd.read_parquet(ROOT / "external/hf/detectood_ood/data/processed/exp31/dolci_sft_first_user.parquet", columns=["source_dataset", "first_user"])
sft_src = sft_src[sft_src["first_user"].notna()].reset_index(drop=True)["source_dataset"]
def prefix(pid):
    pid = re.sub(r"-request-.*$", "", pid); pid = re.sub(r"[-_]\d+[a-z0-9]*$", "", pid); return re.sub(r"[-_][0-9a-f]{8,}$", "", pid)
src = np.where(meta.part.values == "sft", sft_src.reindex(meta.row.values).values, None).astype(object)
d = meta.part.values == "dpo"; src[d] = ["dpo:" + prefix(p) for p in meta.prompt_id.values[d]]
WGM = ["WildGuardMix", "dpo:tulu_v3.9_synthetic_finalresp_wildguardmixtrain_decontaminated"]
WJ = ["WildJailbreak", "dpo:tulu_v3.9_wildjailbreak_decontaminated"]
CCN = ["CoCoNot", "dpo:tulu-3-sft-coconot-regenerated"]
r3 = np.isin(src, WGM + WJ + CCN); rng = np.random.default_rng(0)
rc = np.zeros(len(src), bool); rc[rng.choice(len(src), int(r3.sum()), replace=False)] = True
ARMS = {"R0": np.zeros(len(src), bool), "R1": np.isin(src, WGM), "R2": np.isin(src, WJ), "R3": r3, "Rc": rc}
for a, rm in ARMS.items(): log(a, "rows removed", int(rm.sum()))
json.dump({a: int(rm.sum()) for a, rm in ARMS.items()} | {"rc_overlap_with_r3": int((rc & r3).sum())}, open(OUT / "arms.json", "w"), indent=1)
m = np.load(IDX / "mix.npy").astype(np.float32); mh = np.load(IDX / "mix_hash.npy")
t = np.load(IDX / "traffic.npy").astype(np.float32); th = np.load(IDX / "traffic_hash.npy")
it = faiss.IndexFlatIP(384); it.add(t); n_t = len(t); log("traffic index")
pool = pd.read_json(OUT / "pool.jsonl", lines=True); q = t[pool.row.values]; qh = th[pool.row.values]
st, jj = it.search(q, K + EXTRA); st = np.where(th[jj] == qh[:, None], -np.inf, st)
def score(removed):
    sub = np.where(~removed)[0]; im = faiss.IndexFlatIP(384); im.add(m[sub]); n_m = len(sub)
    sm, ii = im.search(q, K + EXTRA); ii = sub[ii]; sm = np.where(mh[ii] == qh[:, None], -np.inf, sm)
    near = sm.max(1); near_src = src[ii[np.arange(len(q)), sm.argmax(1)]]
    sims = np.concatenate([st, sm], 1); side = np.concatenate([np.zeros_like(st), np.ones_like(sm)], 1)
    top = np.argsort(-sims, 1, kind="stable")[:, :K]; valid = np.isfinite(np.take_along_axis(sims, top, 1))
    nm = (np.take_along_axis(side, top, 1) * valid).sum(1); nt = valid.sum(1) - nm
    return ((nt + 1) / n_t) / ((nm + 1) / n_m), near, near_src
rows = []
for a, rm in ARMS.items():
    ratio, near, ns = score(rm); log(a, "scored")
    for k in range(len(pool)): rows.append({"id": pool.id.iat[k], "arm": a, "ratio": float(ratio[k]), "distance": float(1 - near[k]), "nearest_sim": float(near[k]), "nearest_source": ns[k]})
S = pd.DataFrame(rows).merge(pool.drop(columns=["text"]), on="id"); S.to_parquet(OUT / "scores.parquet"); log("done", len(S))
