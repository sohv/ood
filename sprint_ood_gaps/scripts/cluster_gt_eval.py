"""Cluster ground truth step 2 (designs/cluster_gt.md): normalised Dolci share per topic, traffic-only labels at 3 cuts, and
AUROC / precision@30 / precision@100 of each topic detector against traffic-only, under the MiniLM and the Qwen ground truth.
Bootstrap over topics (2,000 resamples) for intervals. Outputs runs/cluster_gt/{topics_gt.csv, eval.csv, summary.json}, figs/cluster_gt/."""
import json
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
ROOT = Path(__file__).resolve().parents[1]; OUT = ROOT / "runs/cluster_gt"; FIG = ROOT / "figs/cluster_gt"; FIG.mkdir(parents=True, exist_ok=True)
c = pd.read_csv(OUT / "topic_counts.csv").set_index("topic")
tp = pd.read_csv(ROOT / "runs/rare_gaps/topics.csv").set_index("lab")
c = c.loc[tp.index]  # the 1,998 topics with >= 50 members
NT, ND = c.traffic_n.sum(), 1_710_578
for e in ("minilm", "qwen"):
    a = c[f"dolci_{e}"] / ND; b = c.traffic_n / NT; c[f"share_{e}"] = a / (a + b)
CUTS = (0.02, 0.05, 0.10)
det = pd.DataFrame(index=tp.index)
det["distance_v1 (MiniLM)"] = tp.median_score
det["gap_score (ratio)"] = tp.median_his
J = {r["topic"]: 100 - r["grade"]["coverage"] for r in map(json.loads, open(ROOT / "runs/judge_best/primary_scores.jsonl")) if r.get("grade")}
det["judge (100-coverage)"] = pd.Series(J)
v2 = pd.read_csv(ROOT / "runs/detector_v2/scores.csv").set_index("topic")
det["detector_v2 b_c"] = v2.b_c; det["detector_v2 two_stage"] = v2.two_stage
det["distance (Qwen)"] = pd.read_csv(ROOT / "runs/robust/topics_both.csv", index_col=0).qwen
rng = np.random.default_rng(0); rows = []
def prec(y, s, k): return y[np.argsort(-s, kind="stable")[:k]].mean()
for e in ("minilm", "qwen"):
    for cut in CUTS:
        y_all = (c[f"share_{e}"] < cut).values
        for d in det.columns:
            s_all = det[d].values; ok = ~np.isnan(s_all); y, s = y_all[ok], s_all[ok]
            B = [rng.integers(0, len(y), len(y)) for _ in range(2000)]
            au = [roc_auc_score(y[i], s[i]) for i in B if 0 < y[i].sum() < len(i)]
            rows.append({"gt": e, "cut": cut, "detector": d, "n_topics": int(ok.sum()), "traffic_only": int(y.sum()), "base_rate": y.mean(),
                         "auroc": roc_auc_score(y, s), "auroc_lo": np.quantile(au, .025), "auroc_hi": np.quantile(au, .975),
                         "p@30": prec(y, s, 30), "p@100": prec(y, s, 100),
                         "p@100_lo": np.quantile([prec(y[i], s[i], 100) for i in B[:500]], .025), "p@100_hi": np.quantile([prec(y[i], s[i], 100) for i in B[:500]], .975)})
E = pd.DataFrame(rows); E.to_csv(OUT / "eval.csv", index=False)
gt = c.copy()
for e in ("minilm", "qwen"):
    gt[f"label_{e}"] = np.where(gt[f"share_{e}"] < 0.05, "traffic_only", np.where(gt[f"share_{e}"] > 0.5, "dolci_heavy", "shared"))
gt.to_csv(OUT / "topics_gt.csv")
summ = {"n_topics": len(c), "traffic_n": int(NT), "dolci_n": ND,
        "label_counts": {e: gt[f"label_{e}"].value_counts().to_dict() for e in ("minilm", "qwen")},
        "traffic_only_counts": {e: {str(k): int((c[f"share_{e}"] < k).sum()) for k in CUTS} for e in ("minilm", "qwen")},
        "agreement_traffic_only_005": {"both": int(((c.share_minilm < .05) & (c.share_qwen < .05)).sum()), "minilm_only": int(((c.share_minilm < .05) & (c.share_qwen >= .05)).sum()),
                                       "qwen_only": int(((c.share_minilm >= .05) & (c.share_qwen < .05)).sum())},
        "spearman_share_minilm_qwen": float(c.share_minilm.corr(c.share_qwen, method="spearman"))}
(OUT / "summary.json").write_text(json.dumps(summ, indent=1)); print(json.dumps(summ, indent=1))
pd.set_option("display.width", 250); print(E[E.cut == 0.05].round(3).to_string(index=False))
P = E[E.cut == 0.05]; dets = list(det.columns); x = np.arange(len(dets))
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
for ax, (m, lo, hi, yl, base) in zip(axes, [("auroc", "auroc_lo", "auroc_hi", "AUROC, traffic-only vs rest", 0.5), ("p@100", "p@100_lo", "p@100_hi", "precision@100 (share traffic-only)", None)]):
    for j, (e, col) in enumerate((("minilm", "#3b6ea8"), ("qwen", "#c77b30"))):
        q = P[P["gt"] == e].set_index("detector").loc[dets]; xx = x + (j - .5) * .38
        ax.bar(xx, q[m], .38, color=col, label=f"{e} ground truth (traffic-only {int(q.traffic_only.iat[0])} of {int(q.n_topics.iat[0])})")
        ax.errorbar(xx, q[m], yerr=[q[m] - q[lo], q[hi] - q[m]], fmt="none", ecolor="black", capsize=2, lw=.8)
        if m == "p@100": ax.axhline(q.base_rate.iat[0], color=col, ls=":", lw=1)
    if base: ax.axhline(base, c="0.4", ls="--", lw=.8)
    ax.set_xticks(x); ax.set_xticklabels([d.replace(" (", "\n(").replace("v2 ", "v2\n") for d in dets], fontsize=7); ax.set_title(yl, fontsize=9)
axes[0].legend(frameon=False, fontsize=7); axes[0].set_ylim(0, 1)
fig.suptitle("Topic detectors vs cluster ground truth (Dolci prompts assigned to nearest traffic-topic centroid), Olmo-3 traffic", fontsize=10)
fig.text(.5, .005, "1,998 exp37 topics; traffic-only = normalised Dolci share < 0.05; 95% bootstrap over topics; dotted = base rate. MiniLM truth vs MiniLM distance is circular.", ha="center", fontsize=7, color="0.35")
fig.tight_layout(rect=(0, .03, 1, .95)); fig.savefig(FIG / "detectors_vs_cluster_gt.png", dpi=150)
