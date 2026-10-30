"""Part 16 post-hoc check (exploratory, chosen after the main result): do the groupings see a planted removal at all?
Per family and grouping: share of removed prompts in groups of their kind; how many of the 5 groups whose Dolci share drops most are of
that kind (vs 20 random removals); and the traffic-minus-Dolci share of all groups of that kind before and after removal.
Writes runs/gap_taxonomy/grouping_recovery/visible_check.json."""
import json, random
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]; D = ROOT / "runs/gap_taxonomy"
z = np.load(D / "emergent/clusters.npz", allow_pickle=True); ids = [str(x) for x in z["ids"]]
G = {"desc": dict(zip(ids, z["desc_fine_s0"].tolist())), "raw": dict(zip(ids, z["raw_fine_s0"].tolist()))}
kinds = json.load(open(D / "grouping_recovery/group_kinds.json"))["kinds"]; A = json.load(open(D / "removal_arms.json")); fam = A["family"]
tra = [i for i in A["traffic"] if i in G["desc"]]; allD = [i for i in A["dolci_arms"]["full"] if i in G["desc"]]
share = lambda g, v: np.bincount([G[g][i] for i in v], minlength=100) / len(v)
out = {}
for F in ("safety", "code", "multilingual"):
    rem = [i for i in allD if fam.get(i) == F]; others = [i for i in allD if fam.get(i) != F]
    for g in ("desc", "raw"):
        isF = np.array([F in kinds.get(f"{g}:{k}", []) for k in range(100)]); d0 = share(g, allD); dm = share(g, others); drop = d0 - dm
        nulls = []
        for s in range(1, 21):
            rr = set(random.Random(s).sample(others, len(rem))); dr = d0 - share(g, [i for i in allD if i not in rr]); nulls.append(int(isF[np.argsort(-dr)[:5]].sum()))
        tF = share(g, tra)[isF].sum()
        out.setdefault(F, {})[g] = {"removed_in_F_groups": round(float(isF[[G[g][i] for i in rem]].mean()), 3),
            "top5_drop_groups_of_kind": int(isF[np.argsort(-drop)[:5]].sum()), "random_median": float(np.median(nulls)), "largest_drop_pp": round(float(drop.max() * 100), 1),
            "traffic_share_F_groups": round(float(tF), 3), "dolci_full": round(float(d0[isF].sum()), 3), "dolci_minus": round(float(dm[isF].sum()), 3),
            "diff_full": round(float(tF - d0[isF].sum()), 3), "diff_minus": round(float(tF - dm[isF].sum()), 3)}
json.dump(out, open(D / "grouping_recovery/visible_check.json", "w"), indent=1); print(json.dumps(out, indent=1))
