"""figs/topic_split/detector_candidates.png: test AUROC minus one-sided tf-idf for the step 5 candidates on each topic split
(runs/topic_split/improve_summary.json)."""
import json
from pathlib import Path
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
ROOT = Path(__file__).resolve().parents[1]; S = json.load(open(ROOT / "runs/topic_split/improve_summary.json"))
parts = [p for p in ["lsa", "lsa_grouped", "minilm", "minilm_grouped"] if p in S]
names = {"cand_a_rankavg_minilm_tfidf_act": "(a) rank-avg MiniLM+tf-idf+activations", "cand_d_rankavg_minilm_tfidf": "(d) rank-avg MiniLM+tf-idf",
         "maha_pca256_l20": "(b) Mahalanobis PCA-256, layer 20", "v1_knn_kth_1": "MiniLM kNN k=1", "act": "activation kNN k=1 (layer for (a))"}
fig, ax = plt.subplots(figsize=(11, 4.2)); w = 0.16
for i, key in enumerate(names):
    xs, ys, lo, hi = [], [], [], []
    for j, p in enumerate(parts):
        k = S[p]["activation_layer_for_a"] if key == "act" else key
        r = S[p]["results"].get(k)
        if r is None: continue
        xs.append(j + (i - 2) * w); ys.append(r["test_minus_tfidf"]); lo.append(r["test_minus_tfidf"] - r["diff_lo"]); hi.append(r["diff_hi"] - r["test_minus_tfidf"])
    ax.bar(xs, ys, w, yerr=[lo, hi], capsize=2, label=names[key])
ax.axhline(0, color="k", lw=0.8)
ax.set_xticks(range(len(parts))); ax.set_xticklabels([f"{p}\n({S[p]['bootstrap_unit']} bootstrap){' - primary' if p == 'lsa' else ''}" for p in parts], fontsize=8)
ax.set_ylabel("test AUROC minus one-sided tf-idf"); ax.legend(fontsize=7, ncol=2, loc="lower left")
ax.set_title("Candidate one-sided detectors vs one-sided tf-idf on the topic split test (Olmo-3-7B activations, MiniLM)", fontsize=10)
fig.text(0.5, 0.005, "Topic-level AUROC on the test half; 95% paired bootstrap intervals (1,000 resamples). (c) concatenated-layer kNN dropped (over the 20-min rule).",
         ha="center", fontsize=7.5); fig.tight_layout(rect=(0, 0.03, 1, 1)); fig.savefig(ROOT / "figs/topic_split/detector_candidates.png", dpi=150)
