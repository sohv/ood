"""Part 12, HypotheSAEs secondary measure (design point 5, done as in scripts/gap_instag_shift.py): all 20 selected features per arm,
ranked by the package's signed separation score (rank 1 = most traffic-over-Dolci). Each distinct description is judged once, blind to
arm, with scripts/gap_removal_judge.py's question by Sonnet 5.5 and by Astra. Per family F and judge: best rank of an F-description and
the highest separation score among F-descriptions, in full, minus_F and random_F. Writes runs/gap_taxonomy/hypothesaes/shift.json."""
import json, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]; O = ROOT / "runs/gap_taxonomy/hypothesaes"; sys.path.insert(0, str(ROOT / "scripts"))
import gap_removal_judge as J
arms = ["full"] + [f"{p}_{F}" for F in J.KINDS for p in ("minus", "random")]
D = {a: pd.read_csv(O / f"{a}.csv").sort_values("target_separation_score", ascending=False).reset_index(drop=True) for a in arms}
descs = sorted({d for a in arms for d in D[a].interpretation})
items = [{"k": k, "desc": d} for k, d in enumerate(descs)]; batches = [items[i:i + 20] for i in range(0, len(items), 20)]
out = {"n_descriptions": len(descs), "judges": {}}
for name, fn, workers in [("sonnet", J.sonnet, 4), ("astra", J.astra, 3)]:
    lab = {}
    with ThreadPoolExecutor(workers) as ex:
        for r in ex.map(fn, batches): lab.update(r or {})
    fam = {F: {descs[k] for k, v in lab.items() if F in v} for F in J.KINDS}
    res = {"unjudged": len(descs) - len(lab), "f_descriptions": {F: sorted(v) for F, v in fam.items()}, "families": {}}
    for F in J.KINDS:
        row = {}
        for a in ("full", f"minus_{F}", f"random_{F}"):
            hit = D[a][D[a].interpretation.isin(fam[F])]
            row[a] = {"best_rank_of_20": int(hit.index.min()) + 1 if len(hit) else None,
                      "max_separation_score": round(float(hit.target_separation_score.max()), 3) if len(hit) else None}
        res["families"][F] = row
    out["judges"][name] = res
json.dump(out, open(O / "shift.json", "w"), indent=1)
print(json.dumps({j: v["families"] for j, v in out["judges"].items()}, indent=1))
