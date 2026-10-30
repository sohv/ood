"""Part 21 ranking and null: NBER topic categories (runs/gap_taxonomy/nber_topic/tags.json) ranked by traffic share minus Dolci share,
per Dolci arm, as in scripts/review_part15_null20.py (same 20 random same-size removals from other families, seed 0). Per family: best rank
of its categories (code: computer_programming / Technical Help; safety and multilingual: none) and of translation / Writing (nearest
category, reported alongside), for full, minus_F, stored random_F and the 20 randoms; and (amendment) one statistic per arm: the largest
drop in any category's Dolci share against full Dolci, with minus_F counted as having moved a category only if its maximum exceeds the
maximum of all 20 randoms. Fine (24) and coarse (7) levels. Writes runs/gap_taxonomy/nber_topic/summary.json."""
import json, sys
from collections import Counter
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from gap_nber_topic import COARSE, OUT, D
arms = json.load(open(D / "removal_arms.json")); fam = arms["family"]; full = list(arms["dolci_arms"]["full"])
tags = {i: c for i, c in json.load(open(OUT / "tags.json")).items() if c}
to_coarse = {c: k for k, v in COARSE.items() for c in v}
LEVELS = {"fine": lambda c: c, "coarse": lambda c: to_coarse[c]}
KIND = {"fine": {"code": ["computer_programming"], "safety": [], "multilingual": []},
        "coarse": {"code": ["Technical Help"], "safety": [], "multilingual": []}}
NEAR = {"fine": "translation", "coarse": "Writing"}
out = {"n_tagged": len(tags), "n_traffic": sum(i in tags for i in arms["traffic"]), "n_dolci_full": sum(i in tags for i in full)}
for lvl, f in LEVELS.items():
    tr = [i for i in arms["traffic"] if i in tags]; ct = Counter(f(tags[i]) for i in tr)
    def ranks(dolci):
        do = [i for i in dolci if i in tags]; cd = Counter(f(tags[i]) for i in do)
        sc = sorted(((ct[x] / len(tr) - cd[x] / len(do), x) for x in ct), key=lambda p: -p[0])
        return {x: r for r, (_, x) in enumerate(sc, 1)}, {x: round(100 * d, 2) for d, x in sc}
    def dshare(dolci):
        do = [i for i in dolci if i in tags]; cd = Counter(f(tags[i]) for i in do); return {x: cd[x] / len(do) for x in set(map(f, tags.values()))}
    def maxdrop(dolci):
        a = dshare(dolci); x = max(a, key=lambda k: sh_full[k] - a[k]); return round(100 * (sh_full[x] - a[x]), 2), x
    rk_full, diff_full = ranks(full); out[f"{lvl}/full_diff_points"] = diff_full; sh_full = dshare(full)
    for F in ["safety", "code", "multilingual"]:
        n_rem = len(full) - len(arms["dolci_arms"][f"minus_{F}"]); others = [i for i in full if fam[i] != F]
        rng = np.random.default_rng(0); rnd, rmax = [], []
        for _ in range(20):
            d = set(rng.choice(others, n_rem, replace=False).tolist()); arm = [i for i in full if i not in d]; rnd.append(ranks(arm)[0]); rmax.append(maxdrop(arm))
        rk_m, diff_m = ranks(arms["dolci_arms"][f"minus_{F}"]); rk_s = ranks(arms["dolci_arms"][f"random_{F}"])[0]
        cell = {}
        for name, cats in (("kind", KIND[lvl][F]), ("nearest", [NEAR[lvl]])):
            if not cats: cell[name] = None; continue
            b = lambda rk: min(rk.get(x, 999) for x in cats); rr = [b(r) for r in rnd]
            cell[name] = {"categories": cats, "full": b(rk_full), "minus": b(rk_m), "stored_random": b(rk_s), "random20": rr,
                          "randoms_strictly_worse": int(sum(r > b(rk_m) for r in rr)), "found": bool(sum(r > b(rk_m) for r in rr) >= 19),
                          "diff_points_full": diff_full.get(cats[0]), "diff_points_minus": diff_m.get(cats[0])}
        m = maxdrop(arms["dolci_arms"][f"minus_{F}"])
        cell["max_drop"] = {"minus_points": m[0], "minus_category": m[1], "random20_points": [r[0] for r in rmax],
                            "random20_categories": dict(Counter(r[1] for r in rmax)), "moved": bool(m[0] > max(r[0] for r in rmax))}
        out[f"{lvl}/{F}"] = cell
json.dump(out, open(OUT / "summary.json", "w"), indent=1)
print(json.dumps(out, indent=1))
