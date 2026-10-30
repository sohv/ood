"""Step 4 of designs/one_sided_negatives.md (as amended after review): unseen same-kind sets scored with one-sided MiniLM kNN.
Reference = Dolci minus the held-out cov_* rows and every exact copy (via mix_hash[mix_idx]). Threshold = 95th percentile of
pooled held-out Dolci. Writes runs/negatives_one_sided/{rows.parquet, summary.csv}."""
import glob, json
from pathlib import Path
import numpy as np, pandas as pd, faiss
from statsmodels.stats.proportion import proportion_confint

ROOT = Path(__file__).resolve().parents[1]; IDX = ROOT / "runs/negatives/index"; OUT = ROOT / "runs/negatives_one_sided"
CLOSE = {"codealpaca", "codefeedback", "magicoder", "metamath", "mathinstruct", "orca_math", "ifeval_like_argilla", "beavertails",
         "harmbench", "jbb_harmful", "oasst2", "zebralogic", "knights_knaves", "tasksource", "toxic_chat"}
FAR = {"dolly", "no_robots", "sciq", "webinstruct"}; FORMAT = {"toolace", "glaive_fc", "spider", "hermes_fc"}

mh = np.load(IDX / "mix_hash.npy")
cov = [json.loads(l) for f in sorted(glob.glob(str(ROOT / "runs/negatives/sets/cov_*.jsonl"))) for l in open(f)]
cov = pd.DataFrame(cov)[["set", "family", "mix_idx", "text"]]
held_hash = set(mh[cov.mix_idx.values].tolist())
ref = np.flatnonzero(~np.isin(mh, list(held_hash)))
M = np.load(IDX / "mix.npy").astype(np.float32); faiss.normalize_L2(M); faiss.omp_set_num_threads(8)
Mc = M[cov.mix_idx.values].copy()
ix = faiss.IndexFlatIP(384); ix.add(M[ref]); del M

sets = pd.read_parquet(OUT / "sets_emb.parquet"); sets = sets[~sets.member].reset_index(drop=True)
E = np.stack(sets.emb.values).astype(np.float32); faiss.normalize_L2(E)
s_cov, _ = ix.search(Mc, 5); s_set, _ = ix.search(E, 5)
cov["k1"] = 1 - s_cov[:, 0]; cov["mean5"] = 1 - s_cov.mean(1); cov["near_copy"] = s_cov[:, 0] >= 0.95
sets["k1"] = 1 - s_set[:, 0]; sets["mean5"] = 1 - s_set.mean(1); sets["near_copy"] = s_set[:, 0] >= 0.95
thr = {d: float(np.percentile(cov[d], 95)) for d in ["k1", "mean5"]}
sets["tag"] = ["close" if s in CLOSE else "far" if s in FAR else "format" if s in FORMAT else "reference" for s in sets.set]
pd.concat([cov.assign(kind="heldout_dolci", tag="heldout"), sets.drop(columns="emb")]).to_parquet(OUT / "rows.parquet")

out = []
for name, g in list(sets.groupby("set")) + [("heldout_dolci_pooled", cov)]:
    r = {"set": name, "tag": g["tag"].iloc[0] if "tag" in g else "heldout", "n": len(g), "near_copy_share": round(g.near_copy.mean(), 3)}
    for d in ["k1", "mean5"]:
        k = int((g[d] > thr[d]).sum()); lo, hi = proportion_confint(k, len(g), method="wilson")
        r[f"flag_{d}"] = round(k / len(g), 3); r[f"lo_{d}"] = round(lo, 3); r[f"hi_{d}"] = round(hi, 3)
        r[f"fails_{d}"] = bool(r["tag"] == "close" and lo > 0.10)
    out.append(r)
s = pd.DataFrame(out).sort_values(["tag", "flag_k1"]); s.to_csv(OUT / "summary.csv", index=False)
fam = cov.groupby("family").apply(lambda g: pd.Series({"n": len(g), "flag_k1": (g.k1 > thr["k1"]).mean(), "near_copy": g.near_copy.mean()})).round(3)
fam.to_csv(OUT / "heldout_by_family.csv")
json.dump({"thresholds_95pct_heldout": thr, "n_reference": int(len(ref)), "n_heldout": len(cov)}, open(OUT / "meta.json", "w"), indent=1)
pd.set_option("display.width", 200); print(thr); print(s.to_string(index=False))
