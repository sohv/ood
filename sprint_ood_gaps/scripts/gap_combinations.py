"""Combination gaps (designs/gap_taxonomy.md part 13): pairs of part 8 properties (plus task type) that occur together in traffic much more
than in Dolci, beyond what each property alone shows. Uses part 8's tags only (no new calls); properties with kappa < 0.4 vs Sole are left out.
Pair ratio = (traffic share with both + 0.5/n) / (Dolci share with both + 0.5/n); lift = pair ratio / max(single ratios). Pairs with >= 10
traffic prompts. Writes runs/gap_taxonomy/facets/combinations.{json,md}."""
import json, itertools, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import gap_facets as F
D = F.D; S = json.load(open(D / "summary.json"))
ok = [k for k, v in S["facets"].items() if v["kappa_sole"] is not None and v["kappa_sole"] >= 0.4]
R = {r["id"]: r for r in F.rows()}; T = F._load(D / "tags.jsonl")
feat = lambda t: {k for k in ok if t.get(k)} | {"task=" + str(t.get("task"))}
tr = [feat(T[i]) for i in T if R[i]["src"] == "traffic"]; do = [feat(T[i]) for i in T if R[i]["src"] == "dolci"]
nt, nd = len(tr), len(do); names = sorted(set().union(*tr, *do))
ratio = lambda a, b: (a + 0.5) / nt / ((b + 0.5) / nd)
single = {x: ratio(sum(x in s for s in tr), sum(x in s for s in do)) for x in names}
rows = []
for a, b in itertools.combinations(names, 2):
    if a.startswith("task=") and b.startswith("task="): continue
    ct = sum(a in s and b in s for s in tr); cd = sum(a in s and b in s for s in do)
    if ct < 10: continue
    r = ratio(ct, cd); rows.append({"pair": [a, b], "traffic": ct, "dolci": cd, "traffic_share": round(ct / nt, 4), "dolci_share": round(cd / nd, 4),
                                    "pair_ratio": round(r, 2), "single_ratios": [round(single[a], 2), round(single[b], 2)], "lift": round(r / max(single[a], single[b]), 2)})
rows.sort(key=lambda x: -x["lift"])
json.dump({"properties_used": ok, "n_traffic": nt, "n_dolci": nd, "pairs": rows}, open(D / "combinations.json", "w"), indent=1)
md = [f"Pairs with >= 10 traffic prompts: {len(rows)}. Sorted by lift (pair ratio over the larger single ratio).", "",
      "| pair | traffic n | Dolci n | pair ratio | single ratios | lift |", "|---|---|---|---|---|---|"]
md += [f"| {x['pair'][0]} + {x['pair'][1]} | {x['traffic']} | {x['dolci']} | {x['pair_ratio']} | {x['single_ratios']} | {x['lift']} |" for x in rows[:25]]
(D / "combinations.md").write_text("\n".join(md) + "\n"); print("\n".join(md))
