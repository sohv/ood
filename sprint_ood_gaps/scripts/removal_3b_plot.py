"""figs/removal_3b/removal_3b.png: AUROC of the score change for each removal (MiniLM, tf-idf) against 20 random removals of the same size."""
import json
from pathlib import Path
import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
ROOT = Path(__file__).resolve().parents[1]; S = json.load(open(ROOT / "runs/removal_3b/summary.json"))
rows = [("safety", "topic_auroc_change", "random_topic_auroc", "safety (topics, 21 harmful)"),
        ("coding", "topic_auroc_change", "random_topic_auroc", "coding (topics, 268)"),
        ("aya_only", "prompt_auroc_change", "random_prompt_auroc", "Aya removed (prompts)"),
        ("all_non_english", "prompt_auroc_change", "random_prompt_auroc", "all non-English removed (prompts)")]
fig, ax = plt.subplots(figsize=(10, 4.2)); cols = {"minilm": "#3b6fb6", "tfidf": "#d07a2b"}
for i, (k, real, rnd, lab) in enumerate(rows):
    for j, det in enumerate(["minilm", "tfidf"]):
        v = S[f"{det}/{k}"]; x = i + (j - 0.5) * 0.3
        ax.scatter(np.full(len(v[rnd]), x) + np.random.default_rng(0).uniform(-0.05, 0.05, len(v[rnd])), v[rnd], s=10, color="grey", alpha=0.6, zorder=2)
        ax.scatter([x], [v[real]], s=70, color=cols[det], zorder=3, label=("MiniLM distance" if det == "minilm" else "one-sided tf-idf") if i == 0 else None)
ax.axhline(0.5, color="grey", ls=":", lw=0.8); ax.set_xticks(range(4)); ax.set_xticklabels([r[3] for r in rows], fontsize=8.5)
ax.set_ylabel("AUROC of score change, removed kind vs rest"); ax.set_ylim(0.3, 1.0); ax.legend(fontsize=8, loc="upper right")
ax.set_title("Removal test with training-only detectors (Olmo-3 Dolci; traffic queries 25 per topic)", fontsize=10)
fig.text(0.5, 0.005, "Coloured dots: the kind removed. Grey dots: 20 random removals of the same size. Kind labels: Gemini grader (topics); py3langid (prompts).", ha="center", fontsize=7.5)
fig.tight_layout(rect=(0, 0.04, 1, 1)); fig.savefig(ROOT / "figs/removal_3b/removal_3b.png", dpi=150)
