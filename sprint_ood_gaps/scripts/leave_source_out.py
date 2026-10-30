"""Leave-one-source-out planted gaps (designs/leave_source_out.md). CPU faiss, float32 arithmetic.
For each held-out Dolci source S: remove S's mix rows, score S's covered anchors (positives) and 100 anchors per other
covered set (negatives) against the reduced mix and the full traffic pool. Writes runs/leave_source_out/rows.parquet
and summary.csv."""
import glob, json, re, time
from pathlib import Path
import numpy as np, pandas as pd, faiss
from sklearn.metrics import roc_auc_score

import sys
TRAFFIC = sys.argv[1] if len(sys.argv) > 1 else "all"   # "all" or "lmsys" (LMSYS-only traffic, Sole's check 2026-10-23)
ROOT = Path(__file__).resolve().parents[1]; IDX = ROOT / "runs/negatives/index"; OUT = ROOT / ("runs/leave_source_out" if TRAFFIC == "all" else f"runs/leave_source_out_{TRAFFIC}")
OUT.mkdir(parents=True, exist_ok=True)
K, EXTRA = 50, 20
faiss.omp_set_num_threads(8)
def log(*a): print(time.strftime("%H:%M:%S"), *a, flush=True)

SOURCES = {
    "tool_use": ["Dolci Instruct Tool Use"],
    "tablegpt": ["TableGPT", "dpo:tulu_v3.9_table_gpt"],
    "sciriff": ["SciRiff", "dpo:tulu_v3.9_sciriff"],
    "aya": ["Aya", "dpo:tulu_v3.9_aya"],
    "coconot": ["CoCoNot", "dpo:tulu-3-sft-coconot-regenerated"],
    "logic_puzzles": ["Logic Puzzles"],
    "precise_if": ["Dolci Instruct Precise IF", "dpo:IF_sft_data_verified_permissive"],
    "openassistant": ["OpenAssistant", "dpo:oasst1_converted"],
    "wildchat": ["Wildchat", "dpo:filtered_wc_sample", "dpo:Wildchat-1M-gpt-4.1-regenerated-english", "dpo:Wildchat-1m-gpt-4.1-regeneration-not-english"],
}

# source label per mix row
meta = pd.read_parquet(IDX / "mix_meta.parquet")
sft_src = pd.read_parquet(ROOT / "external/hf/detectood_ood/data/processed/exp31/dolci_sft_first_user.parquet", columns=["source_dataset", "first_user"])
sft_src = sft_src[sft_src["first_user"].notna()].reset_index(drop=True)["source_dataset"]
def prefix(pid):
    pid = re.sub(r"-request-.*$", "", pid); pid = re.sub(r"[-_]\d+[a-z0-9]*$", "", pid); return re.sub(r"[-_][0-9a-f]{8,}$", "", pid)
src = np.where(meta.part.values == "sft", sft_src.reindex(meta.row.values).values, None).astype(object)
dmask = meta.part.values == "dpo"
src[dmask] = ["dpo:" + prefix(p) for p in meta.prompt_id.values[dmask]]
for k, v in SOURCES.items():
    n = int(np.isin(src, v).sum()); log(f"{k}: {n} mix rows"); assert n > 0, k

m = np.load(IDX / "mix.npy").astype(np.float32); mh = np.load(IDX / "mix_hash.npy")
t = np.load(IDX / "traffic.npy").astype(np.float32); th = np.load(IDX / "traffic_hash.npy")
if TRAFFIC == "lmsys":
    keep_t = np.array([json.loads(l)["src"] in ("lmsys", "lmsys_ext") for l in open(IDX / "traffic_rows.jsonl")])
    t, th = t[keep_t], th[keep_t]; log("LMSYS-only traffic rows:", len(t))
it = faiss.IndexFlatIP(384); it.add(t); n_t = len(t)
log("traffic index built")

# anchors: embeddings from the set files, olmo_len from the scored files
anc = []
for f in sorted(glob.glob(str(ROOT / "runs/negatives/sets/cov_*.jsonl"))):
    lens = {}
    sf = ROOT / "runs/negatives/scores" / Path(f).name
    if sf.exists():
        for l in open(sf):
            r = json.loads(l); lens[r["mix_idx"]] = r.get("olmo_len")
    for l in open(f):
        r = json.loads(l); anc.append({"set": r["set"], "mix_idx": r["mix_idx"], "emb": r["emb"], "olmo_len": lens.get(r["mix_idx"])})
anc = pd.DataFrame(anc); anc["source"] = src[anc.mix_idx.values]
log("anchors", len(anc), "from", anc.set.nunique(), "sets")
rng = np.random.default_rng(0)

def score(q, qh, keep):
    sub = np.where(keep)[0]
    im = faiss.IndexFlatIP(384); im.add(m[sub]); n_m = len(sub)
    sm, ii = im.search(q, K + EXTRA); ii = sub[ii]
    sm = np.where(mh[ii] == qh[:, None], -np.inf, sm)
    st, jj = it.search(q, K + EXTRA)
    st = np.where(th[jj] == qh[:, None], -np.inf, st)
    near = sm.max(1)
    sims = np.concatenate([st, sm], 1); side = np.concatenate([np.zeros_like(st), np.ones_like(sm)], 1)
    top = np.argsort(-sims, 1, kind="stable")[:, :K]
    valid = np.isfinite(np.take_along_axis(sims, top, 1))
    nm = (np.take_along_axis(side, top, 1) * valid).sum(1); nt = valid.sum(1) - nm
    return ((nt + 1) / n_t) / ((nm + 1) / n_m), near

rows, summ = [], []
full_keep = np.ones(len(m), bool)
for name, labels in SOURCES.items():
    pos = anc[anc.source.isin(labels)]
    pos = pos.iloc[rng.choice(len(pos), min(1000, len(pos)), replace=False)]
    neg = pd.concat([g.sample(min(100, len(g)), random_state=0) for _, g in anc[~anc.source.isin(labels)].groupby("set")])
    q = np.array(pos.emb.tolist() + neg.emb.tolist(), dtype=np.float32); qh = mh[np.r_[pos.mix_idx.values, neg.mix_idx.values]]
    keep = ~np.isin(src, labels)
    ratio, near = score(q, qh, keep)
    ratio_full, near_full_all = score(q, qh, full_keep); near_full = near_full_all[: len(pos)]
    y = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
    lens = np.r_[pos.olmo_len.values, neg.olmo_len.values].astype(float)
    res = {"held_out": name, "n_pos": len(pos), "n_neg": len(neg), "mix_rows_removed": int((~keep).sum()),
           "pos_nearest_sim_before": float(np.median(near_full)), "pos_nearest_sim_after": float(np.median(near[: len(pos)])),
           "neg_nearest_sim": float(np.median(near[len(pos):]))}
    res["auroc_ratio_before_removal"] = float(roc_auc_score(y, ratio_full)); res["auroc_one_sided_before_removal"] = float(roc_auc_score(y, 1 - near_full_all))
    res["auroc_short_len"] = float(roc_auc_score(y[np.isfinite(lens)], -lens[np.isfinite(lens)]))
    for sname, s in (("one_sided", 1 - near), ("ratio", ratio), ("olmo_len", lens)):
        ok = np.isfinite(s)
        res[f"auroc_{sname}"] = float(roc_auc_score(y[ok], s[ok]))
        thr = np.quantile(s[len(pos):][np.isfinite(s[len(pos):])], 0.98)
        res[f"recall_at_2pctFPR_{sname}"] = float((s[: len(pos)] > thr).mean())
    summ.append(res); log(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in res.items()}))
    for k in range(len(q)):
        rows.append({"held_out": name, "is_pos": int(y[k]), "set": (pos.set.values.tolist() + neg.set.values.tolist())[k],
                     "mix_idx": int(np.r_[pos.mix_idx.values, neg.mix_idx.values][k]), "ratio": float(ratio[k]), "one_sided": float(1 - near[k]), "olmo_len": lens[k]})
pd.DataFrame(rows).to_parquet(OUT / "rows.parquet"); s = pd.DataFrame(summ); s.to_csv(OUT / "summary.csv", index=False)
print(s[["held_out", "pos_nearest_sim_before", "pos_nearest_sim_after", "auroc_one_sided_before_removal", "auroc_one_sided", "auroc_ratio_before_removal", "auroc_ratio", "auroc_short_len", "recall_at_2pctFPR_one_sided", "recall_at_2pctFPR_ratio"]].round(3).to_string(index=False))
