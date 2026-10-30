"""Part 21 figure: rank (by traffic share minus Dolci share) of the category each removal moves most, NBER topic taxonomy, for full Dolci,
the planted removal and 20 random removals. Reads runs/gap_taxonomy/nber_topic/{tags.json} and removal_arms.json (recomputes ranks)."""
import json, sys
from collections import Counter
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).resolve().parent))
from gap_nber_topic import COARSE, OUT, D
ROOT = D.parents[1]; a = json.load(open(D / "removal_arms.json")); full = a["dolci_arms"]["full"]; fam = a["family"]
tags = {i: c for i, c in json.load(open(OUT / "tags.json")).items() if c}; co = {c: k for k, v in COARSE.items() for c in v}
CELLS = [("code", "computer_programming", "fine"), ("code", "Technical Help", "coarse"), ("multilingual", "specific_info", "fine"), ("safety", "how_to_advice", "fine")]
def rank(dolci, cat, lvl):
    f = (lambda c: c) if lvl == "fine" else co.__getitem__
    tr = [i for i in a["traffic"] if i in tags]; ct = Counter(f(tags[i]) for i in tr); do = [i for i in dolci if i in tags]; cd = Counter(f(tags[i]) for i in do)
    sc = sorted(ct, key=lambda x: -(ct[x] / len(tr) - cd[x] / len(do))); return sc.index(cat) + 1, len(sc)
fig, ax = plt.subplots(figsize=(7, 3.4)); Ns = []
for y, (F, cat, lvl) in enumerate(CELLS):
    others = [i for i in full if fam[i] != F]; n = len(full) - len(a["dolci_arms"][f"minus_{F}"]); rng = np.random.default_rng(0); rr = []
    for _ in range(20):
        d = set(rng.choice(others, n, replace=False).tolist()); rr.append(rank([i for i in full if i not in d], cat, lvl)[0])
    (r_full, N), r_min = rank(full, cat, lvl), rank(a["dolci_arms"][f"minus_{F}"], cat, lvl)[0]
    ax.scatter(rr, [y] * 20, color="grey", alpha=0.35, s=30, label="20 random removals" if y == 0 else None)
    ax.scatter([r_full], [y], marker="|", s=250, color="black", label="full Dolci" if y == 0 else None)
    ax.scatter([r_min], [y], color="#c0602a", s=50, zorder=3, label="planted removal" if y == 0 else None)
    Ns.append(N)
ax.set_yticks(range(len(CELLS)), [f"{F} removed: {cat} ({lvl}, of {N})" for (F, cat, lvl), N in zip(CELLS, Ns)], fontsize=8.5)
ax.set_xlim(26.5, 0.5); ax.set_xlabel("rank by traffic share minus Dolci share (1 = most traffic-heavy)"); ax.legend(fontsize=7.5, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3)
ax.set_title("Rank of the most-moved NBER topic category after each planted removal", fontsize=10)
fig.text(0.5, -0.1, "Chatterji et al. 2025 topic classifier, labels by Claude Sonnet 5.5; 1,989 traffic and 1,991 Dolci prompts; randoms from other families, seed 0",
         ha="center", fontsize=7)
fig.tight_layout(); fig.savefig(ROOT / "figs/gap_taxonomy/nber_topic_removal.png", dpi=150, bbox_inches="tight")
