"""Figure: strict (Sole) vs exp49 poor-answer effect, member minus control per topic, distance_only and random arms."""
import json
from pathlib import Path
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
ROOT = Path(__file__).resolve().parents[1]
st = json.load(open(ROOT / "runs/judge/strict/summary_gpt-6.1-sol.json")); ex = json.load(open(ROOT / "runs/behaviour_distance/graded/summary.json"))["poor"]
rows = [("distance-only, exp49 grader", ex["distance_only"]), ("distance-only, strict (Sole)", st["distance_only"]),
        ("distance-only, strict, no truncated", st["distance_only_no_truncated"]), ("distance-only, strict, no Dolci-member prompts", st["distance_only_no_dolci_members"]),
        ("random, exp49 grader", ex["random"]), ("random, strict (Sole)", st["random"])]
fig, ax = plt.subplots(figsize=(8.5, 3.8))
for k, (n, r) in enumerate(rows):
    ax.errorbar(r["effect"], k, xerr=[[r["effect"] - r["ci95"][0]], [r["ci95"][1] - r["effect"]]], fmt="o", color="#2a78d6" if "strict" in n else "#777", capsize=3)
ax.set_yticks(range(len(rows)), [f"{n} (n={r['n_topics']})" for n, r in rows]); ax.invert_yaxis()
ax.axvline(0, color="k", lw=0.8); ax.axvline(0.05, color="#b5462f", lw=0.8, ls="--")
ax.set_xlabel("poor-answer rate, member minus matched control (mean over topics)")
ax.set_title("Poor answers on harmless requests, distance-flagged vs matched control prompts (Olmo-3-7B-Instruct-DPO)\n10 replies per side per topic; topic bootstrap 95% CI; dashed line = pre-set 0.05 bar", fontsize=9)
fig.tight_layout(); fig.savefig(ROOT / "figs/judge/strict_poor_effect.png", dpi=150, bbox_inches="tight")
