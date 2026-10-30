"""Figure for part 16: for each removed kind, traffic-minus-Dolci share of that kind's description groups, on full Dolci, after the
kind's removal, and after 20 random removals of the same size (mean and range). Writes figs/gap_taxonomy/planted_vs_natural.png."""
import json, random
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
ROOT = Path(__file__).resolve().parents[1]; D = ROOT / "runs/gap_taxonomy"
z = np.load(D / "emergent/clusters.npz", allow_pickle=True); ids = [str(x) for x in z["ids"]]; G = dict(zip(ids, z["desc_fine_s0"].tolist()))
kinds = json.load(open(D / "grouping_recovery/group_kinds.json"))["kinds"]; A = json.load(open(D / "removal_arms.json")); fam = A["family"]
tra = [i for i in A["traffic"] if i in G]; allD = [i for i in A["dolci_arms"]["full"] if i in G]
labels = {"safety": "safety", "code": "code", "multilingual": "non-English"}
fig, ax = plt.subplots(figsize=(8, 4.5)); x = np.arange(3); w = 0.26; res = {}
for j, F in enumerate(("safety", "code", "multilingual")):
    isF = {k for k in range(100) if F in kinds.get(f"desc:{k}", [])}; sh = lambda v: 100 * np.mean([G[i] in isF for i in v])
    rem = [i for i in allD if fam.get(i) == F]; others = [i for i in allD if fam.get(i) != F]
    full = sh(tra) - sh(allD); minus = sh(tra) - sh(others)
    rnd = [sh(tra) - sh([i for i in allD if i not in set(random.Random(s).sample(others, len(rem)))]) for s in range(1, 21)]
    res[F] = {"full": full, "minus": minus, "random_mean": float(np.mean(rnd)), "random_min": float(min(rnd)), "random_max": float(max(rnd))}
    for off, v, c, lab in ((-w, full, "#9e9e9e", "full Dolci"), (0, np.mean(rnd), "#6baed6", "after a random removal (20, range)"), (w, minus, "#d62728", "after removing this kind")):
        b = ax.bar(j + off, v, w, color=c, label=lab if j == 0 else None)
        ax.text(j + off, v + (0.6 if v >= 0 else -1.6), f"{v:+.1f}", ha="center", fontsize=8)
    ax.errorbar(j, np.mean(rnd), yerr=[[np.mean(rnd) - min(rnd)], [max(rnd) - np.mean(rnd)]], color="k", capsize=3, lw=1)
ax.axhline(0, color="k", lw=0.6); ax.set_xticks(x); ax.set_xticklabels([labels[F] for F in ("safety", "code", "multilingual")])
ax.set_ylabel("traffic share minus Dolci share (points)\nhigher = more traffic-heavy")
ax.set_title("Share of prompts in groups of the removed kind: traffic minus Dolci (Olmo-3-7B Dolci vs WildChat+LMSYS)\n"
             "100 description-based groups; 2,000 traffic and 2,000 Dolci prompts; removals of 105 / 397 / 99 Dolci prompts", fontsize=8.5)
ax.set_ylim(-18, 17); ax.legend(fontsize=8, loc="lower right"); fig.tight_layout(); fig.savefig(ROOT / "figs/gap_taxonomy/planted_vs_natural.png", dpi=160)
json.dump(res, open(D / "grouping_recovery/planted_vs_natural.json", "w"), indent=1); print(json.dumps(res, indent=1))
