"""Figure for Part 18: share "real gap" per arm and judge with Wilson 95% intervals, from runs/loop_labels/llm/summary.json."""
import json
from pathlib import Path
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
ROOT = Path(__file__).resolve().parents[1]; s = json.load(open(ROOT / "runs/loop_labels/llm/summary.json"))
fig, ax = plt.subplots(figsize=(6, 3.6))
for k, (j, c) in enumerate((("sonnet", "#3a6ea5"), ("astra", "#c0602a"))):
    for x, arm in enumerate(("loop", "random")):
        a = s[j][arm]; p = a["real_share"]; lo, hi = a["wilson95"]
        ax.errorbar(x + (k - 0.5) * 0.18, p, yerr=[[p - lo], [hi - p]], fmt="o", color=c, capsize=4, label=j.capitalize() if x == 0 else None)
        ax.annotate(f'{a["real_gap"]}/{a["judged"]}', (x + (k - 0.5) * 0.18 + 0.05, p), fontsize=8, va="center")
ax.axhline(0.5, ls=":", c="grey", lw=1); ax.set_xticks([0, 1], ["loop gaps (n=30)", "random topics (n=20)"]); ax.set_xlim(-0.5, 1.5); ax.set_ylim(0, 1)
ax.set_ylabel('share "real gap" (Wilson 95%)'); ax.legend(frameon=False, fontsize=8)
ax.set_title('LLM-judged "real gap" share, corrected loop vs random topics', fontsize=10)
fig.text(0.5, -0.02, "Olmo-3-7B gap list, Dolci neighbours (2 MiniLM + 2 tf-idf); judges see the blind label page (arm hidden); LLM labels only", ha="center", fontsize=7.5)
fig.tight_layout(); fig.savefig(ROOT / "figs/loop_labels_llm_provisional.png", dpi=150, bbox_inches="tight")
