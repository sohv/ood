"""Part 16 (designs/gap_taxonomy.md): do description clusters recover planted gaps better than raw-text (topic-like) clusters?
Groups fixed from part 10 (clusters.npz, seed 0). Group kinds judged blind once (Sonnet, part 12's question) from names.jsonl.
Arms: minus_F from removal_arms.json, and 20 random removals of the same count from the other families (seeds 1-20). With and without
part 15's set-aside. Writes runs/gap_taxonomy/grouping_recovery/{group_kinds.json, summary.json}."""
import json, os, random, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]; D = ROOT / "runs/gap_taxonomy"; O = D / "grouping_recovery"; O.mkdir(exist_ok=True)
sys.path.insert(0, str(ROOT / "scripts"))
import gap_removal_judge as J, gap_facets as F
z = np.load(D / "emergent/clusters.npz", allow_pickle=True); ids = [str(x) for x in z["ids"]]
G = {"desc": dict(zip(ids, z["desc_fine_s0"].tolist())), "raw": dict(zip(ids, z["raw_fine_s0"].tolist()))}
names = [json.loads(l) for l in open(D / "emergent/names.jsonl")]; names = [n for n in names if n["level"] == "fine" and n.get("name")]
if not (O / "group_kinds.json").exists():
    if os.environ.get("ALLOW_JUDGE_CALLS") != "1": sys.exit(f"missing {O / 'group_kinds.json'}; recreating it calls Sonnet. Set ALLOW_JUDGE_CALLS=1 to do that on purpose.")
    items = [{"k": k, "desc": f'{n["name"]}: {n.get("summary") or ""}', "arm": n["arm"], "node": n["node"]} for k, n in enumerate(names)]
    random.Random(0).shuffle(items)
    lab = {}
    for b in [items[i:i + 20] for i in range(0, len(items), 20)]: lab.update(J.sonnet(b))
    kinds = {f'{it["arm"]}:{it["node"]}': lab.get(it["k"], []) for it in items}
    json.dump({"kinds": kinds, "unjudged": len(items) - len(lab)}, open(O / "group_kinds.json", "w"), indent=1)
kinds = json.load(open(O / "group_kinds.json"))["kinds"]
A = json.load(open(D / "removal_arms.json")); fam = A["family"]; T = F._load(D / "facets/tags.jsonl")
noisy = lambda i: i in T and bool(T[i].get("placeholder_or_artefact") or T[i].get("templated_dataset_item"))
tra = [i for i in A["traffic"] if i in G["desc"]]; allD = [i for i in A["dolci_arms"]["full"] if i in G["desc"]]
def measure(g, F_, dol, setaside):
    tr = [i for i in tra if not (setaside and noisy(i))]; do = [i for i in dol if not (setaside and noisy(i))]
    ct = np.bincount([G[g][i] for i in tr], minlength=100) / max(len(tr), 1); cd = np.bincount([G[g][i] for i in do], minlength=100) / max(len(do), 1)
    order = np.argsort(-(ct - cd)); isF = np.array([F_ in kinds.get(f"{g}:{k}", []) for k in range(100)])
    rank = {int(k): r for r, k in enumerate(order, 1)}
    return {"top10_F": int(isF[order[:10]].sum()), "best_F_rank": min((rank[k] for k in range(100) if isF[k]), default=101), "n_F_groups": int(isF.sum())}
out = {}
for F_ in ("safety", "code", "multilingual"):
    rem = [i for i in allD if fam.get(i) == F_]; others = [i for i in allD if fam.get(i) != F_]; removed = set(rem)
    minus = [i for i in allD if i not in removed]
    rands = []
    for s in range(1, 21):
        rr = set(random.Random(s).sample(others, len(rem))); rands.append([i for i in allD if i not in rr])
    for g in ("desc", "raw"):
        for sa in (False, True):
            m = measure(g, F_, minus, sa); R = [measure(g, F_, r, sa) for r in rands]
            med_top = float(np.median([r["top10_F"] for r in R])); better = sum(m["best_F_rank"] < r["best_F_rank"] for r in R)
            rec = (m["top10_F"] - med_top >= 2) and better >= 19
            out.setdefault(F_, {})[f"{g}{'_setaside' if sa else ''}"] = {"minus": m, "random_top10_F_median": med_top,
                "random_best_rank_median": float(np.median([r["best_F_rank"] for r in R])), "randoms_beaten_on_best_rank": better, "recovered": rec}
json.dump(out, open(O / "summary.json", "w"), indent=1)
print("family | grouping | F-groups in top 10 (minus vs random median) | best F rank (minus vs random median) | randoms beaten /20 | recovered")
for F_, v in out.items():
    for g, x in v.items():
        print(F_, g, f'{x["minus"]["top10_F"]} vs {x["random_top10_F_median"]}', f'{x["minus"]["best_F_rank"]} vs {x["random_best_rank_median"]}', x["randoms_beaten_on_best_rank"], x["recovered"], f'(F groups: {x["minus"]["n_F_groups"]})')
