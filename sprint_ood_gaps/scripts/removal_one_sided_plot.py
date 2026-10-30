"""figs/removal_one_sided/removal_recover.png from runs/removal_one_sided/summary.json"""
import json
from pathlib import Path
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
ROOT = Path(__file__).resolve().parents[1]; S = json.load(open(ROOT / "runs/removal_one_sided/summary.json"))
kinds = list(S); fig, ax = plt.subplots(1, 2, figsize=(11, 4))
for i, det in enumerate(["k1", "mean5"]):
    g = [S[k][det]["gain"] for k in kinds]; r = [sum(S[k][det]["random_gains"]) / 3 for k in kinds]
    ax[0].bar([j + (i - 0.5) * 0.4 for j in range(3)], g, 0.2, label=f"kind removed, kNN {det}")
    ax[0].bar([j + (i - 0.5) * 0.4 + 0.2 for j in range(3)], r, 0.2, label=f"random removal (mean of 3), {det}", alpha=0.5)
    a = [S[k][det]["topic_auroc_change"] for k in kinds]; ar = [max(S[k][det]["topic_auroc_change_random"]) for k in kinds]
    ax[1].bar([j + (i - 0.5) * 0.4 for j in range(3)], a, 0.2, label=f"kind removed, {det}")
    ax[1].bar([j + (i - 0.5) * 0.4 + 0.2 for j in range(3)], ar, 0.2, label=f"random removal (max of 3), {det}", alpha=0.5)
for a_, t, yl in [(ax[0], "Gain in kind topics among the top 308", "topics of the removed kind entering the top 308"),
                  (ax[1], "Topic AUROC of the score change, kind vs other topics", "AUROC")]:
    a_.set_xticks(range(3)); a_.set_xticklabels([f"{k}\n({S[k]['n_kind_topics']} topics)" for k in kinds]); a_.set_title(t, fontsize=10); a_.set_ylabel(yl, fontsize=9)
ax[1].axhline(0.5, color="grey", ls=":", lw=0.8); ax[1].set_ylim(0, 1); ax[0].legend(fontsize=7); ax[1].legend(fontsize=7, loc="lower right")
fig.suptitle("One-sided MiniLM distance to Dolci, removal of a kind vs random removal (Olmo-3 Dolci, 1,998 traffic topics)", fontsize=10)
fig.text(0.5, 0.005, "25 non-member prompts per topic, topic score = median; kind labels from the partner's Gemini grader; random removals of the same size, 3 seeds",
         ha="center", fontsize=7.5); fig.tight_layout(rect=(0, 0.03, 1, 0.95)); fig.savefig(ROOT / "figs/removal_one_sided/removal_recover.png", dpi=150)
