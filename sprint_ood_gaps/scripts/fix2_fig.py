"""Figure for the eval-only fix2 run: refusal of harmless requests on held-out NAME_k gap topics, member vs matched covered control, per model."""
import json
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np
ROOT = Path(__file__).resolve().parents[1]; S = json.load(open(ROOT / "runs/fix2/graded/summary.json"))
BLUE, ORANGE, NEUTRAL = "#2a78d6", "#eb6834", "#b9b9b3"
fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
for ax, (s, title) in zip(axes, (("main", "12 NAME_k gap topics (ours)"), ("fix_overrefused", "20 NAME_k fix_overrefused targets"))):
    x = np.arange(3); w = 0.38
    for j, (role, col) in enumerate((("member", ORANGE), ("control", NEUTRAL))):
        ax.bar(x + (j - .5) * w, [S[s][f"effect_{m}"][role] for m in ("base", "control", "fix")], w, color=col, label="gap-topic prompts" if role == "member" else "matched covered prompts")
    ax.set_xticks(x, ["untrained", "control adapter", "NAME_1 fix adapter"]); ax.set_title(f"{title}, n prompts {20 if s == 'main' else 10}/topic", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
axes[0].set_ylabel("refusal of harmless requests\n(safety_grader_v2 grader; 2 replies/prompt)"); axes[1].legend(frameon=False, fontsize=9, loc="upper right")
fig.suptitle("Olmo-3-7B-Instruct-DPO: refusal of harmless held-out prompts on NAME_k gap topics, by adapter")
fig.tight_layout(); fig.savefig(ROOT / "figs/fix2/name_topics_refusal.png", dpi=150); print(ROOT / "figs/fix2/name_topics_refusal.png")
