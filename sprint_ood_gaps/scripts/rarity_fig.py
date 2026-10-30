"""Figure for runs/rarity_check/summary.json: refusal rate by gap-score quintile, confirmed over-refusal vs grader-only, NAME_k and template prompts removed."""
import json
from pathlib import Path
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
ROOT = Path(__file__).resolve().parents[1]; s = json.load(open(ROOT / "runs/rarity_check/summary.json"))
a = s["20k_confirmed_over_refusal (primary)"]["ratio"]["rate_by_score_quintile"]; b = s["20k_grader_refused_harmless (same prompts as primary)"]["ratio"]["rate_by_score_quintile"]
fig, ax = plt.subplots(figsize=(6.4, 4)); x = range(1, 6)
ax.plot(x, [100 * v for v in b], "s--", color="#999999", label="grader says harmless and Olmo refused (includes explicit / jailbreak)")
ax.plot(x, [100 * v for v in a], "o-", color="#d95f02", label="over-refusal confirmed by both harm readers")
ax.set_xticks(list(x)); ax.set_xlabel("gap-score quintile (5 = rarest in Dolci)"); ax.set_ylabel("% of prompts")
ax.set_title("Refusal of harmless-graded prompts by rarity in Dolci, Olmo-3-7B-Instruct-DPO", fontsize=10)
fig.text(0.5, 0.005, "20k run, NAME_k and toxic-template prompts removed (17,479 prompts, 121 confirmed over-refusals);\ncluster Spearman with gap score 0.10 (0.01-0.19) confirmed vs 0.18 (0.09-0.27) grader-only", ha="center", fontsize=7)
ax.legend(fontsize=7, frameon=False); fig.tight_layout(rect=(0, 0.07, 1, 1)); (ROOT / "figs/rarity").mkdir(parents=True, exist_ok=True); fig.savefig(ROOT / "figs/rarity/refusal_by_rarity.png", dpi=150)
