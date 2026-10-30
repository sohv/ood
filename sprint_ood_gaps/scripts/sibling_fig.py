"""Figure for designs/sibling_split.md: change in a sub-topic's median distance after moving a sibling, random prompts, or half of itself to Dolci."""
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[1]; d = pd.read_csv(ROOT / "runs/sibling_split/pairs.csv")
BLUE, ORANGE, NEUTRAL = "#2a78d6", "#eb6834", "#b9b9b3"
order = [1579, 956, 257, 25, 1646, 1530]; names = {1579: "1579 run Vicuna", 956: "956 'are you Vicuna'", 257: "257 sexual topics", 25: "25 install X", 1646: "1646 troubleshoot", 1530: "1530 'X vs Y'"}
g = d.groupby("topic")[["d_sibling_dist", "d_control_dist", "d_self_dist"]].mean().loc[order]
fig, ax = plt.subplots(figsize=(9, 4.4)); x = np.arange(len(order)); w = 0.27
for j, (c, col, lab) in enumerate((("d_self_dist", BLUE, "half of the sub-topic itself (reference)"), ("d_sibling_dist", ORANGE, "a sibling sub-topic"), ("d_control_dist", NEUTRAL, "same number of random prompts"))):
    ax.bar(x + (j - 1) * w, g[c], w, color=col, label=lab)
ax.axvline(2.5, color="#666", lw=0.8, ls=":")
ax.set_xticks(x, [names[c] + ("\n(gap-score top)" if c in (1579, 956, 257) else "\n(distance top)") for c in order], fontsize=8); ax.axhline(0, color="black", lw=0.6)
ax.set_ylabel("change in median distance to Dolci\n(mean over 6 ordered sub-topic pairs)"); ax.legend(frameon=False, fontsize=8.5, title="moved into Dolci:", title_fontsize=8.5, loc="lower right")
ax.spines[["top", "right"]].set_visible(False)
ax.set_title("Change in a sub-topic's distance score after moving prompts into Dolci (MiniLM, 6 topics split in 3 by Claude)", fontsize=9.5)
fig.tight_layout(); fig.savefig(ROOT / "figs/sibling_split/distance_change.png", dpi=150); print(ROOT / "figs/sibling_split/distance_change.png")
