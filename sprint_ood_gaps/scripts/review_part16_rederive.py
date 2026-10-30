"""Independent re-derivation of part 16's headline numbers (review item 10, 30 Oct), written from the input files only, without
reading scripts/gap_grouping_*.py. For each grouping (desc_fine_s0, raw_fine_s0 from runs/gap_taxonomy/emergent/clusters.npz) and
family F (safety, code, multilingual): traffic share minus Dolci share, in points, of the groups judged to be of kind F
(group_kinds.json), on full Dolci, after removing F's Dolci prompts (removal_arms.json minus_F), and after 20 random removals of the
same size drawn from Dolci prompts of other families (seed 0 here, so randoms differ from the original). Also: how many of the 5 groups
with the largest drop in Dolci share are of kind F, for the F removal and for each random removal. Shares are over sample prompts
that have a group (part 8 sample). Writes runs/review/part16_rederive.json."""
import json
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]; G = ROOT / "runs/gap_taxonomy"
z = np.load(G / "emergent/clusters.npz"); ids = list(z["ids"])
arms = json.load(open(G / "removal_arms.json")); kinds = json.load(open(G / "grouping_recovery/group_kinds.json"))["kinds"]
fam = arms["family"]; full = list(arms["dolci_arms"]["full"])
out = {}
for gname, key in [("desc", "desc_fine_s0"), ("raw", "raw_fine_s0")]:
    grp = dict(zip(ids, z[key].tolist())); ng = int(z[key].max()) + 1
    def shares(sample):  # share of the sample in each group, among sample prompts that have a group
        g = [grp[i] for i in sample if i in grp]; return np.bincount(g, minlength=ng) / len(g), len(g)
    t_sh, nt = shares(arms["traffic"]); d_full, nd = shares(full)
    for F in ["safety", "code", "multilingual"]:
        kg = [g for g in range(ng) if F in kinds.get(f"{gname}:{g}", [])]
        diff = lambda d: 100 * (t_sh[kg].sum() - d[kg].sum())
        def top5(d):  # kind groups among the 5 groups whose Dolci share drops most
            drop = d_full - d; top = np.argsort(-drop, kind="stable")[:5]; return int(sum(g in kg for g in top))
        removed = set(full) - set(arms["dolci_arms"][f"minus_{F}"]); d_minus, _ = shares(arms["dolci_arms"][f"minus_{F}"])
        others = [i for i in full if fam[i] != F]; rng = np.random.default_rng(0); rd, rt = [], []
        for _ in range(20):
            drop_ids = set(rng.choice(others, len(removed), replace=False).tolist())
            d_r, _ = shares([i for i in full if i not in drop_ids]); rd.append(diff(d_r)); rt.append(top5(d_r))
        out[f"{gname}/{F}"] = {"n_kind_groups": len(kg), "n_traffic": nt, "n_dolci_full": nd, "n_removed": len(removed),
                               "removed_in_kind_groups": round(float(np.mean([grp[i] in kg for i in removed if i in grp])), 3),
                               "traffic_share": round(100 * float(t_sh[kg].sum()), 2), "dolci_full": round(100 * float(d_full[kg].sum()), 2),
                               "dolci_minus": round(100 * float(d_minus[kg].sum()), 2),
                               "diff_full": round(diff(d_full), 2), "diff_minus": round(diff(d_minus), 2),
                               "diff_random_mean": round(float(np.mean(rd)), 2), "diff_random_min": round(float(min(rd)), 2), "diff_random_max": round(float(max(rd)), 2),
                               "top5_kind_minus": top5(d_minus), "top5_kind_random_median": float(np.median(rt)), "top5_kind_random_max": int(max(rt))}
(ROOT / "runs/review").mkdir(exist_ok=True)
json.dump(out, open(ROOT / "runs/review/part16_rederive.json", "w"), indent=1)
for k, v in out.items(): print(k, v)
