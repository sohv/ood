"""Independent re-derivation of part 16's recovery rule (review item 10b, 30 Oct), written without reading scripts/gap_grouping_*.py.
Groups (desc_fine_s0, raw_fine_s0) are ranked by traffic share minus Dolci share, descending (ties keep group order). Per family F and
Dolci arm: number of F-groups (group_kinds.json) in the top 10 and the best rank of an F-group. Rule as briefed: recovered when the
minus_F top-10 count is at least the random median + 2 AND its best rank is strictly better than in at least 19 of 20 randoms. The randoms
are 20 same-size removals from other families' Dolci prompts (seed 0, the same draws as review_part16_rederive.py). Set-aside drops
prompts tagged placeholder_or_artefact or templated_dataset_item (facets/tags.jsonl) from both sides; prompts without tags are kept.
Writes runs/review/part16_recovery.json."""
import json
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]; G = ROOT / "runs/gap_taxonomy"
z = np.load(G / "emergent/clusters.npz"); ids = list(z["ids"])
arms = json.load(open(G / "removal_arms.json")); kinds = json.load(open(G / "grouping_recovery/group_kinds.json"))["kinds"]
fam = arms["family"]; full = list(arms["dolci_arms"]["full"])
tags = {}
for l in open(G / "facets/tags.jsonl"):
    b = json.loads(l)
    for t in b.get("tags") or []:
        if isinstance(t, dict) and isinstance(t.get("i"), int) and 1 <= t["i"] <= len(b["ids"]): tags[b["ids"][t["i"] - 1]] = set(t.get("props") or [])
noisy = {i for i, p in tags.items() if p & {"placeholder_or_artefact", "templated_dataset_item"}}
out = {"n_tagged": len(tags), "n_noisy": len(noisy)}
for gname, key in [("desc", "desc_fine_s0"), ("raw", "raw_fine_s0")]:
    grp = dict(zip(ids, z[key].tolist())); ng = int(z[key].max()) + 1
    for setaside in (False, True):
        keep = lambda s: [i for i in s if i in grp and not (setaside and i in noisy)]
        sh = lambda s: np.bincount([grp[i] for i in keep(s)], minlength=ng) / len(keep(s))
        t_sh = sh(arms["traffic"])
        def stats(dolci, kg):
            order = np.argsort(-(t_sh - sh(dolci)), kind="stable"); rank = {g: r + 1 for r, g in enumerate(order)}
            return sum(rank[g] <= 10 for g in kg), min(rank[g] for g in kg)
        for F in ["safety", "code", "multilingual"]:
            kg = [g for g in range(ng) if F in kinds.get(f"{gname}:{g}", [])]
            n_rem = len(full) - len(arms["dolci_arms"][f"minus_{F}"]); others = [i for i in full if fam[i] != F]
            rng = np.random.default_rng(0); rnd = []
            for _ in range(20):
                d = set(rng.choice(others, n_rem, replace=False).tolist()); rnd.append(stats([i for i in full if i not in d], kg))
            m_top, m_best = stats(arms["dolci_arms"][f"minus_{F}"], kg); f_top, f_best = stats(full, kg)
            r_top_med = float(np.median([r[0] for r in rnd])); beaten = sum(m_best < r[1] for r in rnd)
            out[f"{gname}{'_setaside' if setaside else ''}/{F}"] = {
                "n_kind_groups": len(kg), "full_top10": f_top, "full_best": f_best, "minus_top10": m_top, "minus_best": m_best,
                "random_top10_median": r_top_med, "random_best_median": float(np.median([r[1] for r in rnd])),
                "randoms_beaten_on_best_rank": beaten, "recovered": bool(m_top >= r_top_med + 2 and beaten >= 19)}
json.dump(out, open(ROOT / "runs/review/part16_recovery.json", "w"), indent=1)
for k, v in out.items(): print(k, v)
