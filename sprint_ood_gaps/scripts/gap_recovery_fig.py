"""Slide 9 figure: removed-kind descriptions in each method's top 10, after removing that kind vs after a random removal of the same size,
per method and family, from runs/gap_taxonomy/removal_judge/summary.json (Sonnet judge; Astra in the subtitle). Pre-set recovery bar: +2.
Writes figs/gap_taxonomy/recovery_by_method.png."""
import json
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
ROOT = Path(__file__).resolve().parents[1]; S = json.load(open(ROOT / "runs/gap_taxonomy/removal_judge/summary.json"))
R = S["results"]["sonnet"]; A = S["results"].get("astra", {})
methods = [m for m in ("instag", "hypothesaes", "lloom", "lloom_seeded") if m in R]
names = {"instag": "InsTag\n(10)", "hypothesaes": "HypotheSAEs\n(3-8)", "lloom": "LLooM\n(4-5)", "lloom_seeded": "seeded\nLLooM (2)"}
fams = [("safety", "safety"), ("code", "code"), ("multilingual", "non-English")]
fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), sharey=True)
for ax, (F, lab) in zip(axes, fams):
    x = np.arange(len(methods)); mi = [R[m][F]["minus"] for m in methods]; ra = [R[m][F]["random"] for m in methods]
    ax.bar(x - 0.2, ra, 0.4, color="#6baed6", label="after a random removal"); ax.bar(x + 0.2, mi, 0.4, color="#d62728", label="after removing that kind")
    for i, (a, b) in enumerate(zip(ra, mi)):
        ax.text(i - 0.2, a + 0.1, str(a), ha="center", fontsize=8); ax.text(i + 0.2, b + 0.1, str(b), ha="center", fontsize=8)
        if m_astra := A.get(methods[i], {}).get(F): ax.text(i, -0.75, f"Astra {m_astra['random']}/{m_astra['minus']}", ha="center", fontsize=6.5, color="#555")
    ax.set_xticks(x); ax.set_xticklabels([names[m] for m in methods], fontsize=7.5); ax.set_title(lab, fontsize=10); ax.set_ylim(-1.0, 4)
    ax.axhline(0, color="k", lw=0.5)
axes[0].set_ylabel("descriptions of the removed kind\nin the method's list"); axes[0].legend(fontsize=7.5, loc="upper left")
fig.suptitle("Removed-kind descriptions in each method's traffic-over-Dolci list (list length in brackets) (Olmo-3-7B Dolci vs WildChat+LMSYS)\n"
             "2,000 traffic and 2,000 Dolci prompts; one random removal per family; blind Sonnet judge (Astra counts below bars); pre-set recovery bar: +2", fontsize=8.5)
fig.tight_layout(); fig.savefig(ROOT / "figs/gap_taxonomy/recovery_by_method.png", dpi=160); print("ok", methods)
