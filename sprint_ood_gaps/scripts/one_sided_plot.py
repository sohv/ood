"""Per-family AUROC of one-sided detectors (runs/one_sided/summary.csv) -> figs/one_sided/family_auroc.png"""
from pathlib import Path
import pandas as pd, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
ROOT = Path(__file__).resolve().parents[1]; s = pd.read_csv(ROOT / "runs/one_sided/summary.csv")
dets = {"knn_kth_1": "kNN k=1 (our distance)", "knn_mean_5": "kNN mean of 5", "maha_cc": "class-cond. Mahalanobis", "length": "length"}
fam = s[s.detector == "knn_kth_1"].sort_values(["split", "auroc"])[["family", "split"]]
fig, ax = plt.subplots(figsize=(11, 4.5)); w = 0.2
for i, (d, lab) in enumerate(dets.items()):
    v = s[s.detector == d].set_index("family").loc[fam.family]
    ax.bar([j + (i - 1.5) * w for j in range(len(fam))], v.auroc, w, yerr=[v.auroc - v.lo, v.hi - v.auroc], label=lab, capsize=1.5)
ax.axhline(0.5, color="grey", lw=0.8, ls=":"); ax.axhline(0.95, color="k", lw=0.8, ls="--")
ax.text(len(fam) - 0.5, 0.955, "0.95 (Stefan's expectation)", ha="right", va="bottom", fontsize=8)
ax.set_xticks(range(len(fam))); ax.set_xticklabels([f"{f}\n({sp})" for f, sp in zip(fam.family, fam.split)], fontsize=8)
ax.set_ylabel("AUROC, held-out family vs held-out kept prompts"); ax.set_ylim(0, 1.05); ax.legend(fontsize=8, ncol=4, loc="lower left")
ax.set_title("One-sided OOD detectors on Dolci family hold-out (MiniLM embeddings, Olmo-3 Dolci)", fontsize=10)
fig.text(0.5, 0.005, "1,000 removed-family prompts vs 4,600-5,400 held-out prompts of kept sources per family; reference = rest of Dolci (1.38-1.70M prompts); 95% bootstrap intervals",
         ha="center", fontsize=7.5)
fig.tight_layout(rect=(0, 0.03, 1, 1)); fig.savefig(ROOT / "figs/one_sided/family_auroc.png", dpi=150)
