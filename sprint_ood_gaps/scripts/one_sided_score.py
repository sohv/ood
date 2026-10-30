"""Step 1 of designs/one_sided_negatives.md: Dolci-only family hold-out, one-sided detectors.
For each family F: reference = Dolci minus F minus held-out negatives minus every exact copy (text hash) of either;
score 1,000 F positives and the held-out negatives of other families with kNN distances (MiniLM), Mahalanobis,
class-conditional Mahalanobis and length. Writes runs/one_sided/rows_<family>.parquet and families.json."""
import json, random, re, sys, time
from pathlib import Path
import numpy as np, pandas as pd, faiss
from sklearn.covariance import LedoitWolf

ROOT = Path(__file__).resolve().parents[1]; IDX = ROOT / "runs/negatives/index"; OUT = ROOT / "runs/one_sided"
OUT.mkdir(parents=True, exist_ok=True)
KS = [1, 5, 20, 50]; N_POS, N_NEG, MIN_SRC, N_COV = 1000, 200, 4000, 200_000
FAMILIES = {
    "math": ["Tulu 3 Persona MATH", "Tulu 3 Persona GSM", "Tulu 3 Persona Algebra", "OpenMathInstruct 2", "dpo:tulu-3-sft-personas-math-grade"],
    "code": ["Dolci Instruct Python Algorithms", "Evol CodeAlpaca", "Tulu 3 Persona Python"],
    "safety": ["WildJailbreak", "WildGuardMix", "CoCoNot", "dpo:tulu_v3.9_wildjailbreak_decontaminated",
               "dpo:tulu_v3.9_synthetic_finalresp_wildguardmixtrain_decontaminated", "dpo:tulu-3-sft-coconot-regenerated"],
    "chat": ["Wildchat", "OpenAssistant", "dpo:filtered_wc_sample", "dpo:Wildchat-1M-gpt-4.1-regenerated-english",
             "dpo:Wildchat-1m-gpt-4.1-regeneration-not-english", "dpo:ultrafeedback_cleaned_olmo2", "dpo:DaringAnteater-prefs_olmo2"],
    "tool_use": ["Dolci Instruct Tool Use"],
    "precise_if": ["Dolci Instruct Precise IF", "dpo:IF_sft_data_verified_permissive", "dpo:tulu-3-sft-personas-instruction-following-o3",
                   "dpo:valpy_if_qwq_reasoning_verified_no_reasoning"],
    "reasoning": ["Verifiable Reasoning"],
    "logic": ["Logic Puzzles"],
    "multilingual": ["Aya", "dpo:tulu_v3.9_aya"],
    "flan": ["FLAN"],
    "science": ["Dolci Instruct OpenThoughts3+ Science", "SciRiff", "dpo:tulu_v3.9_sciriff"],
    "table": ["TableGPT", "dpo:tulu_v3.9_table_gpt"],
}


def log(*a): print(time.strftime("%H:%M:%S"), *a, flush=True)


def prefix(pid):
    pid = re.sub(r"-request-.*$", "", pid); pid = re.sub(r"[-_]\d+[a-z0-9]*$", "", pid); return re.sub(r"[-_][0-9a-f]{8,}$", "", pid)


# source label and text hash per mix row (same labelling as scripts/leave_source_out.py)
meta = pd.read_parquet(IDX / "mix_meta.parquet")
sft = pd.read_parquet(ROOT / "external/hf/detectood_ood/data/processed/exp31/dolci_sft_first_user.parquet", columns=["source_dataset", "first_user"])
sft = sft[sft["first_user"].notna()].reset_index(drop=True)
src = np.where(meta.part.values == "sft", sft["source_dataset"].reindex(meta.row.values).values, None).astype(object)
dmask = meta.part.values == "dpo"; src[dmask] = ["dpo:" + prefix(p) for p in meta.prompt_id.values[dmask]]
mh = np.load(IDX / "mix_hash.npy"); n = len(src)
fam_of = np.array([next((f for f, v in FAMILIES.items() if s in v), "") for s in src], dtype=object)
for f in FAMILIES: log(f, int((fam_of == f).sum()), "rows")

# held-out negatives: 200 per source label with >= 4,000 rows, unique texts
rng = np.random.default_rng(0)
vc = pd.Series(src).value_counts(); big = vc[vc >= MIN_SRC].index.tolist()
neg = []
for s in big:
    idx = np.flatnonzero(src == s); idx = idx[~pd.Series(mh[idx]).duplicated().values]
    neg.extend(rng.choice(idx, N_NEG, replace=False).tolist())
neg = np.array(sorted(neg)); neg_hash = set(mh[neg].tolist())
log(f"negatives {len(neg)} from {len(big)} sources")

# dev / test split of families, fixed before scoring
fams = list(FAMILIES); random.Random(0).shuffle(fams); split = {f: ("dev" if i < 6 else "test") for i, f in enumerate(fams)}
json.dump({"split": split, "families": FAMILIES, "neg_sources": big, "n_neg": int(len(neg))}, open(OUT / "families.json", "w"), indent=1)
log("split", split)

# texts for query rows only (length baseline and later reading)
from huggingface_hub import hf_hub_download
dpo_paths = [hf_hub_download("allenai/Dolci-Instruct-DPO", f"data/train-{i:05d}-of-00004.parquet", repo_type="dataset") for i in range(4)]
dpo = pd.concat([pd.read_parquet(p, columns=["chosen"]) for p in dpo_paths], ignore_index=True)["chosen"]
def text(i):
    return sft["first_user"].iat[int(meta.row.iat[i])] if meta.part.iat[i] == "sft" else next(x["content"] for x in dpo.iat[int(meta.row.iat[i])] if x["role"] == "user")

X = np.load(IDX / "mix.npy").astype(np.float32); faiss.normalize_L2(X)
faiss.omp_set_num_threads(8)
only = sys.argv[1:]  # optional family subset
for f in (only or fams):
    t0 = time.time()
    fam_rows = np.flatnonzero(fam_of == f)
    pos_pool = fam_rows[~np.isin(mh[fam_rows], list(neg_hash))]
    pos_pool = pos_pool[~pd.Series(mh[pos_pool]).duplicated().values]
    pos = np.sort(rng.choice(pos_pool, min(N_POS, len(pos_pool)), replace=False))
    negs = neg[fam_of[neg] != f]
    drop_hash = set(mh[fam_rows].tolist()) | neg_hash
    ref = np.flatnonzero(~np.isin(mh, list(drop_hash)))
    log(f"{f}: removed {n - len(ref)} rows, reference {len(ref)}, pos {len(pos)}, neg {len(negs)}")
    q = np.concatenate([pos, negs]); Q = X[q]
    index = faiss.IndexFlatIP(X.shape[1]); index.add(X[ref])
    sims, nn = index.search(Q, max(KS)); del index
    # full-mix nearest similarity of positives, for the validity check (before removal; copies of self excluded)
    base = np.flatnonzero(~np.isin(mh, list(set(mh[q].tolist()))))
    index = faiss.IndexFlatIP(X.shape[1]); index.add(X[base]); sims_full, _ = index.search(Q, 1); del index
    rows = pd.DataFrame({"mix_idx": q, "label": np.r_[np.ones(len(pos), int), np.zeros(len(negs), int)],
                         "source": src[q], "family": fam_of[q], "nn1_sim_before": sims_full[:, 0], "nn1_sim_after": sims[:, 0]})
    for k in KS:
        rows[f"knn_kth_{k}"] = 1 - sims[:, k - 1]; rows[f"knn_mean_{k}"] = 1 - sims[:, :k].mean(1)
    rows["nn_src_after"] = src[ref[nn[:, 0]]]
    # Mahalanobis, shared covariance (Ledoit-Wolf) on a reference sample
    sub = rng.choice(ref, min(N_COV, len(ref)), replace=False); R = X[sub]
    mu = R.mean(0); lw = LedoitWolf().fit(R - mu); P = lw.precision_.astype(np.float32)
    D = Q - mu; rows["maha"] = np.einsum("ij,jk,ik->i", D, P, D)
    # class-conditional: per-source means, pooled within-source covariance, min over sources
    labs = src[sub]; classes = [c for c, m in pd.Series(labs).value_counts().items() if m >= 200]
    keep = np.isin(labs, classes); Rk, Lk = R[keep], labs[keep]
    mus = np.stack([Rk[Lk == c].mean(0) for c in classes])
    resid = Rk - mus[pd.Index(classes).get_indexer(Lk)]
    Pc = LedoitWolf(assume_centered=True).fit(resid).precision_.astype(np.float32)
    dc = np.stack([np.einsum("ij,jk,ik->i", Q - m, Pc, Q - m) for m in mus], 1)
    rows["maha_cc"] = dc.min(1); rows["maha_cc_src"] = np.array(classes, dtype=object)[dc.argmin(1)]
    rows["length"] = [len(text(int(i))) for i in q]
    rows["random"] = rng.random(len(q))
    rows["text"] = [text(int(i))[:2000] for i in q]
    rows.to_parquet(OUT / f"rows_{f}.parquet")
    log(f"{f}: done in {time.time() - t0:.0f}s; pos nn1 before/after {np.median(sims_full[:len(pos), 0]):.3f}/{np.median(sims[:len(pos), 0]):.3f}")
log("all done")
