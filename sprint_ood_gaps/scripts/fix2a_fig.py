"""Figure for branch A fix2: strict poor-answer rate on held-out distance-topic and covered prompts, per model."""
import json
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np
ROOT = Path(__file__).resolve().parents[1]; S = json.load(open(ROOT / "runs/fix2a/graded/summary.json"))
ORANGE, NEUTRAL = "#eb6834", "#b9b9b3"
fig, ax = plt.subplots(figsize=(7.5, 4.4)); x = np.arange(3); w = 0.38
for j, (role, col, lab) in enumerate((("member", ORANGE, "distance-topic prompts (1,050)"), ("control", NEUTRAL, "matched covered prompts (525)"))):
    v = [S["rates"][m][role] for m in ("base", "control", "fix")]
    b = ax.bar(x + (j - .5) * w, v, w, color=col, label=lab)
    ax.bar_label(b, fmt="%.3f", fontsize=8)
ax.set_xticks(x, ["untrained", "covered-data adapter", "gap-data adapter"]); ax.set_ylim(0, 1)
ax.set_ylabel("strict poor-answer rate\n(Sole grader, 1 reply per prompt)"); ax.legend(frameon=False, fontsize=9, loc="upper right")
ax.spines[["top", "right"]].set_visible(False)
ax.set_title("Strict poor-answer rate on held-out prompts, Olmo-3-7B-Instruct-DPO and two LoRA adapters\n"
             "210 distance-only topics; adapters trained on Olmo-3.1-32B answers, 1,546 examples each", fontsize=9.5)
fig.tight_layout(); fig.savefig(ROOT / "figs/fix2/branchA_strict_poor.png", dpi=150); print(ROOT / "figs/fix2/branchA_strict_poor.png")
