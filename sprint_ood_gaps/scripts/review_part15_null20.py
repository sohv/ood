"""Part 15 re-derived and given a 20-random null (review item 12, 30 Oct). Not fully independent: lines 10-33 of scripts/gap_setaside.py
were read earlier for a count check. Features per prompt (facets/tags.jsonl): the part 8 properties with Sole kappa >= 0.4
(facets/summary.json) plus "task=<task>". Features are ranked by traffic share minus Dolci share, descending, over features present in
at least one traffic prompt. Per family F the best rank of F's properties (same lists as part 15) is reported for full, minus_F, the
stored random_F arm, and 20 new same-size random removals from other families' Dolci prompts (seed 0). Set-aside drops prompts tagged
placeholder_or_artefact or templated_dataset_item from both sides. Writes runs/review/part15_null20.json."""
import json
from collections import Counter
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]; G = ROOT / "runs/gap_taxonomy"
arms = json.load(open(G / "removal_arms.json")); fam = arms["family"]; full = list(arms["dolci_arms"]["full"])
S = json.load(open(G / "facets/summary.json"))["facets"]
ok = {k for k, v in S.items() if v["kappa_sole"] is not None and v["kappa_sole"] == v["kappa_sole"] and v["kappa_sole"] >= 0.4}
if not ok: raise SystemExit("facets/summary.json has no Sole kappas (was gap_facets.py analyse run without sole.jsonl?); the feature set would silently shrink to task labels only")
tags = {}
for l in open(G / "facets/tags.jsonl"):
    b = json.loads(l)
    for t in b.get("tags") or []:
        if isinstance(t, dict) and isinstance(t.get("i"), int) and 1 <= t["i"] <= len(b["ids"]): tags[b["ids"][t["i"] - 1]] = t
feat = {i: {p for p in (t.get("props") or []) if p in ok} | {f"task={t.get('task')}"} for i, t in tags.items()}
noisy = {i for i, t in tags.items() if {"placeholder_or_artefact", "templated_dataset_item"} & set(t.get("props") or [])}
FP = {"safety": ["harmful_request", "adversarial_pressure"], "code": ["task=coding", "agentic_or_tool_use"], "multilingual": ["non_english"]}
out = {"n_ok_properties": len(ok)}
for setaside in (False, True):
    keep = lambda s: [i for i in s if i in feat and not (setaside and i in noisy)]
    tr = keep(arms["traffic"]); ct = Counter(x for i in tr for x in feat[i])
    def best(dolci, F):
        do = keep(dolci); cd = Counter(x for i in do for x in feat[i])
        sc = sorted(((ct[x] / len(tr) - cd[x] / len(do), x) for x in ct), key=lambda p: -p[0])
        rank = {x: r for r, (_, x) in enumerate(sc, 1)}; return min(rank.get(x, 999) for x in FP[F]), len(sc), sum(rank.get(x, 999) <= 10 for x in FP[F])
    for F in FP:
        n_rem = len(full) - len(arms["dolci_arms"][f"minus_{F}"]); others = [i for i in full if fam[i] != F]
        rng = np.random.default_rng(0); rnd = []
        for _ in range(20):
            d = set(rng.choice(others, n_rem, replace=False).tolist()); rnd.append(best([i for i in full if i not in d], F))
        m, n, mt = best(arms["dolci_arms"][f"minus_{F}"], F); rt = [r[2] for r in rnd]; rnd = [r[0] for r in rnd]
        out[f"{'setaside' if setaside else 'all'}/{F}"] = {
            "n_features": n, "full": best(full, F)[0], "minus": m, "stored_random": best(arms["dolci_arms"][f"random_{F}"], F)[0],
            "random20": rnd, "random20_median": float(np.median(rnd)), "random20_best": int(min(rnd)),
            "randoms_strictly_worse": int(sum(r > m for r in rnd)), "randoms_tied": int(sum(r == m for r in rnd)),
            "minus_top10": int(mt), "full_top10": int(best(full, F)[2]), "random20_top10_median": float(np.median(rt)), "random20_top10_max": int(max(rt))}
json.dump(out, open(ROOT / "runs/review/part15_null20.json", "w"), indent=1)
for k, v in out.items(): print(k, v)
