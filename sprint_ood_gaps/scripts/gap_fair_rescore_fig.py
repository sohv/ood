"""Part 24 figure: one test for every labelling method (runs/gap_taxonomy/fair_rescore/summary.json). Writes figs/gap_taxonomy/fair_rescore.png."""
import json
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
ROOT = Path(__file__).resolve().parents[1]; S = json.load(open(ROOT / "runs/gap_taxonomy/fair_rescore/summary.json"))
plt.rcParams["font.family"] = ["Ubuntu Sans", "Ubuntu", "Liberation Sans", "DejaVu Sans"]
M = ["our tags", "NBER topics", "moderation classifier", "InsTag", "HypotheSAEs", "LLooM", "LLooM seeded"]; F = ["code", "safety", "multilingual"]; T = {"code": "Code removed", "safety": "Safety removed", "multilingual": "Non-English removed"}
C = {"reach": "#1f66c4", "moved": "#bcd6f5", "not": "#e3e5e9", "none": "#ffffff"}; TC = {"reach": "white", "moved": "#1b2a3d", "not": "#222222", "none": "#8a8f98"}
fig, ax = plt.subplots(figsize=(11.5, 5.6)); ax.set_xlim(0, 3.95); ax.set_ylim(len(M), -0.6); ax.axis("off")
for j, f in enumerate(F): ax.text(0.95 + j + 0.48, -0.3, T[f], ha="center", va="center", fontsize=13, fontweight="bold")
for i, m in enumerate(M):
    ax.text(0.88, i + 0.47, m, ha="right", va="center", fontsize=14)
    for j, f in enumerate(F):
        v = S[m][f]; x = 0.95 + j
        if v.get("no_category"): k, a, b = "none", "no category", ""
        else:
            k = "reach" if v["reaches_list"] else ("moved" if v["found"] else "not")
            rr = f"{v['random_min']}" + (f"-{v['random_max']}" if v["random_max"] != v["random_min"] else "")
            a = f"{v['minus']} of {v['n_categories']}, {v['diff_minus']:+.1f} pts" + ("*" if (m, f) == ("HypotheSAEs", "safety") else ""); b = f"before {v['full']}, randoms {rr}"
        ax.add_patch(plt.Rectangle((x, i + 0.04), 0.96, 0.88, fc=C[k], ec="#c4c8ce", lw=0.8))
        if b: ax.text(x + .48, i + .34, a, ha="center", va="center", fontsize=12, color=TC[k], fontweight="bold"); ax.text(x + .48, i + .68, b, ha="center", va="center", fontsize=10, color=TC[k])
        else: ax.text(x + .48, i + .48, a, ha="center", va="center", fontsize=11.5, color=TC[k])
leg = [("reach", "beats 19+ of 20 randoms and is more common in traffic (95% CI above 0)"), ("moved", "beats 19+ of 20 randoms, but not more common in traffic")]
y = len(M) + 0.14
for k, t in leg:
    ax.add_patch(plt.Rectangle((0.0, y), 0.07, 0.2, fc=C[k], ec="#c4c8ce", lw=0.6, clip_on=False)); ax.text(0.1, y + 0.1, t, va="center", fontsize=11, color="#333333"); y += 0.3
ax.text(0.0, y + 0.1, "Cells: rank after the removal, of how many categories. pts: traffic share minus Dolci share after the removal. *Explicit sexual content neuron, not harmful requests.", va="center", fontsize=10, color="#333333")
fig.savefig(ROOT / "figs/gap_taxonomy/fair_rescore.png", dpi=200, bbox_inches="tight", facecolor="white")
