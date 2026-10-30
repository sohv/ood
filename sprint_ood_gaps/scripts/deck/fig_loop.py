"""Loop diagram for the methods slide. Facts: STATUS.md corrected-loop and loop-negatives entries; runs/loop_onesided/v2/decision.json. Writes figs/gap_taxonomy/loop_diagram.png."""
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
ROOT = Path(__file__).resolve().parents[1].parent
plt.rcParams["font.family"] = ["Ubuntu Sans", "Ubuntu", "Liberation Sans", "DejaVu Sans"]
S = [("Topics", "993k traffic first turns\nMiniLM, k-means\n2,000 topics"), ("Rank", "distance to nearest\nDolci prompt, median\nno traffic labels"), ("Name", "LLM reads 25 prompts,\nnames what they share\n201 topics reviewed"), ("Confirm", "LLM rates coverage:\n10 prompts beside\nnearest Dolci prompt\nconfirm at 50 or below"), ("Merge", "join look-alikes\n\n54 gaps (56 topics)")]
fig, ax = plt.subplots(figsize=(13, 3.8)); ax.set_xlim(0, 13); ax.set_ylim(-0.2, 3.6); ax.axis("off"); w = 2.2; g = 0.45
for i, (h, t) in enumerate(S):
    x = 0.05 + i * (w + g); blue = i == 4
    ax.add_patch(FancyBboxPatch((x, 0.95), w, 2.25, boxstyle="round,pad=0.02,rounding_size=0.08", fc="#1f66c4" if blue else "#eef1f5", ec="#b8c0cc", lw=1))
    c = "white" if blue else "#1b2a3d"; ax.text(x + w / 2, 2.88, h, ha="center", va="center", fontsize=16, fontweight="bold", color=c); ax.text(x + w / 2, 1.85, t, ha="center", va="center", fontsize=10.5, color=c, linespacing=1.3)
    if i < 4: ax.add_patch(FancyArrowPatch((x + w + 0.04, 2.07), (x + w + g - 0.04, 2.07), arrowstyle="-|>", mutation_scale=18, color="#555555", lw=1.5))
x0 = 0.05 + 4 * (w + g) + w / 2; x1 = 0.05 + w / 2
ax.add_patch(FancyArrowPatch((x0, 0.93), (x1, 0.93), connectionstyle="bar,fraction=-0.06", arrowstyle="-|>", mutation_scale=16, color="#999999", lw=1.3, linestyle="--"))
ax.text(6.5, 0.02, "Train on the gaps and repeat: not run in the final version (Neel, 27 Oct: find the data first)", ha="center", va="center", fontsize=12, color="#555555")
fig.savefig(ROOT / "figs/gap_taxonomy/loop_diagram.png", dpi=200, bbox_inches="tight", facecolor="white")
