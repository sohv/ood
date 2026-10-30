"""AUROC per family and detector for runs/one_sided/rows_*.parquet (designs/one_sided_negatives.md step 1).
Higher score = more OOD for every detector. Writes runs/one_sided/summary.csv and prints dev/test means."""
import json
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

ROOT = Path(__file__).resolve().parents[1]; D = ROOT / "runs/one_sided"
split = json.load(open(D / "families.json"))["split"]
B = 300; rng = np.random.default_rng(0)


def recall_at_fpr(y, s, fpr=0.05):
    f, t, _ = roc_curve(y, s); return float(np.interp(fpr, f, t))


out = []
for p in sorted(D.glob("rows_*.parquet")):
    r = pd.read_parquet(p); fam = p.stem[5:]; y = r.label.values
    dets = [c for c in r.columns if c.startswith(("knn_", "maha")) and not c.endswith("_src")] + ["length", "random"]
    band = pd.qcut(r.length.rank(method="first"), 4, labels=False).values
    pi, ni = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    boots = [np.r_[rng.choice(pi, len(pi)), rng.choice(ni, len(ni))] for _ in range(B)]
    for d in dets:
        s = r[d].values; a = roc_auc_score(y, s)
        bs = [roc_auc_score(y[b], s[b]) for b in boots]
        within = [roc_auc_score(y[band == k], s[band == k]) for k in range(4) if 0 < y[band == k].sum() < (band == k).sum()]
        out.append({"family": fam, "split": split[fam], "detector": d, "auroc": a, "lo": np.percentile(bs, 2.5), "hi": np.percentile(bs, 97.5),
                    "auroc_within_length": float(np.mean(within)) if within else np.nan, "recall_at_5fpr": recall_at_fpr(y, s),
                    "pos_nn1_before": float(r.nn1_sim_before[y == 1].median()), "pos_nn1_after": float(r.nn1_sim_after[y == 1].median()),
                    "neg_nn1_after": float(r.nn1_sim_after[y == 0].median()), "n_pos": int(y.sum()), "n_neg": int((1 - y).sum())})
s = pd.DataFrame(out); s.to_csv(D / "summary.csv", index=False)
pd.set_option("display.width", 220)
print(s.pivot_table(index="detector", columns="family", values="auroc").round(3).to_string())
print(s.groupby(["split", "detector"])[["auroc", "auroc_within_length", "recall_at_5fpr"]].mean().round(3).unstack(0).to_string())
print(s.drop_duplicates("family")[["family", "split", "pos_nn1_before", "pos_nn1_after", "neg_nn1_after"]].round(3).to_string(index=False))
