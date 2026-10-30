"""figs/topic_split/topic_split_auroc.png: test AUROC per detector on both topic partitions, and MiniLM kNN AUROC by
how close an OOD topic is to an ID topic (runs/topic_split/summary.json, topics_neighbours.parquet)."""
import json
from pathlib import Path
import pandas as pd, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
ROOT = Path(__file__).resolve().parents[1]; S = json.load(open(ROOT / "runs/topic_split/summary.json"))
dets = {"v1_knn_kth_1": "MiniLM kNN k=1", "v1_knn_mean_5": "MiniLM kNN mean 5", "v1_maha_cc": "MiniLM class-cond. Maha",
        "v1_maha": "MiniLM Mahalanobis", "v1_tfidf": "tf-idf nearest (one-sided)", "v1_length": "length", "random": "random"}
fig, ax = plt.subplots(1, 2, figsize=(12, 4.3), gridspec_kw={"width_ratios": [1.6, 1]})
for i, (p, lab) in enumerate([("minilm", "topics = MiniLM clusters (pipeline_million)"), ("lsa", "topics = tf-idf clusters")]):
    r = [S[p]["detectors"][d]["test"] for d in dets]
    ax[0].bar([j + (i - 0.5) * 0.38 for j in range(len(dets))], [x["auroc"] for x in r], 0.38,
              yerr=[[x["auroc"] - x["lo"] for x in r], [x["hi"] - x["auroc"] for x in r]], capsize=2, label=lab)
ax[0].set_xticks(range(len(dets))); ax[0].set_xticklabels(dets.values(), rotation=25, ha="right", fontsize=8)
ax[0].axhline(0.5, color="grey", ls=":", lw=0.8); ax[0].axhline(0.95, color="k", ls="--", lw=0.8); ax[0].set_ylim(0, 1)
ax[0].set_ylabel("topic-level AUROC, OOD vs ID topics (test half)"); ax[0].legend(fontsize=8, loc="upper right")
ax[0].set_title("One-sided detectors on the topic split test (V1, no Dolci)", fontsize=10)
for p, lab in [("minilm", "MiniLM topics"), ("lsa", "tf-idf topics")]:
    t = pd.read_parquet(ROOT / f"runs/topic_split/{p}/topics_neighbours.parquet"); o = t[t.ood == 1].copy()
    o["bin"] = pd.qcut(o.nearest_id_centroid, 4, labels=False); idt = t[t.ood == 0]
    ax[1].plot(range(4), [roc_auc_score(pd.concat([g, idt]).ood, pd.concat([g, idt]).v1_knn_kth_1) for _, g in o.groupby("bin")], "o-", label=lab)
ax[1].set_xticks(range(4)); ax[1].set_xticklabels(["farthest\nquarter", "2", "3", "nearest\nquarter"], fontsize=8)
ax[1].set_xlabel("OOD topics, by closeness to the nearest ID topic", fontsize=9); ax[1].set_ylabel("AUROC vs all ID topics (MiniLM kNN k=1)", fontsize=9)
ax[1].axhline(0.5, color="grey", ls=":", lw=0.8); ax[1].set_ylim(0.3, 1); ax[1].legend(fontsize=8)
ax[1].set_title("MiniLM kNN AUROC by OOD topic closeness to ID topics", fontsize=10)
fig.text(0.5, 0.005, "Traffic topics split in half by template group; half of topics ID (A half in reference), half OOD; 25 B-half prompts per topic scored. "
         "MiniLM partition 1,912 topics, tf-idf partition 1,719; 95% topic-bootstrap intervals.", ha="center", fontsize=7.5)
fig.tight_layout(rect=(0, 0.04, 1, 1)); fig.savefig(ROOT / "figs/topic_split/topic_split_auroc.png", dpi=150)
