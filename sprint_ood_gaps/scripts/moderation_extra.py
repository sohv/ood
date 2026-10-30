"""Safety-removal amendment 1, extra headlines requested by research-40: AUROC excluding sexual positives (pairs whose positive has no
'sexual' category; 2,000 bootstrap resamples, seed 0) and the category-balanced mean of per-category results for categories with >= 150
positives (from by_category.csv). Output runs/safety_removal/moderation/extra_headlines.json."""
import json
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
D = "runs/safety_removal/moderation"; S = pd.read_parquet(f"{D}/scores.parquet").merge(pd.read_parquet(f"{D}/tfidf.parquet"), on=["id", "arm"])
base = S.drop_duplicates("id").set_index("id"); sex = base.categories.fillna("").str.contains("sexual")
keep_pairs = base[(base.is_pos == 1) & ~sex].pair; ids = base[base.pair.isin(keep_pairs)].index; y = base.loc[ids].is_pos.values
rng = np.random.default_rng(0); pi, ni = np.where(y == 1)[0], np.where(y == 0)[0]; B = [np.r_[rng.choice(pi, len(pi)), rng.choice(ni, len(ni))] for _ in range(2000)]
out = {"excluding_sexual": {"n_pos": int(len(pi))}}
for sc in ("ratio", "distance", "tfidf"):
    W = S.pivot(index="id", columns="arm", values=sc).loc[ids]; a = {k: roc_auc_score(y, W[k]) for k in ("R0", "R3", "Rc")}
    d = [(roc_auc_score(y[b], W.R3.values[b]) - roc_auc_score(y[b], W.Rc.values[b])) for b in B]
    out["excluding_sexual"][sc] = {"R0": a["R0"], "R3": a["R3"], "Rc": a["Rc"], "rise_minus_Rc": a["R3"] - a["Rc"], "lo": float(np.quantile(d, .025)), "hi": float(np.quantile(d, .975))}
C = pd.read_csv(f"{D}/by_category.csv"); C = C[C.n_pos >= 150]
out["category_balanced_mean"] = {"categories": sorted(C.category.unique().tolist())} | {sc: {"R0": float(g.auroc_R0.mean()), "R3": float(g.auroc_R3.mean()), "rise_minus_Rc": float(g.rise_minus_Rc.mean())} for sc, g in C.groupby("score")}
json.dump(out, open(f"{D}/extra_headlines.json", "w"), indent=1); print(json.dumps(out, indent=1))
