"""Adds to runs/gap_taxonomy/property_first/summary.json (labels unchanged): grab-bag counts per arm by two rules, the defining-axis
test far vs random, and per-cluster rows. Rule 1 = defining axis "other" (pre-set axis list). Rule 2 = description says the group has
no common thread (regex below); rule 2 was written after reading the labels, so it is descriptive only."""
import json, re
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import fisher_exact, chi2_contingency
ROOT = Path(__file__).resolve().parents[1]; OUT = ROOT / "runs/gap_taxonomy/property_first"
C = pd.read_parquet(OUT / "clusters.parquet"); S = json.load(open(OUT / "summary.json"))
GRAB = re.compile(r"grab-bag|unrelated|heterogeneous|no (shared|common)|disparate|diverse, unrelated", re.I)
C["grab_bag_rule2"] = C.defining.fillna("").map(lambda s: bool(GRAB.search(s)))
for arm in ["far", "random"]:
    g = C[C.arm == arm]
    S[arm]["grab_bag_rule1_axis_other"] = int((g.defining_axis == "other").sum())
    S[arm]["grab_bag_rule2_description_posthoc"] = int(g.grab_bag_rule2.sum())
    S[arm]["coherent_rule1"] = int((g.defining_axis != "other").sum())
    S[arm]["rows"] = g[["id", "size", "median_distance", "share_in_r50_topics", "defining_axis", "defining", "grab_bag_rule2"]].round(3).to_dict("records")
f, r = C[C.arm == "far"], C[C.arm == "random"]
S["test"] = {"other_far_vs_random_fisher_p": round(float(fisher_exact([[(f.defining_axis == "other").sum(), (f.defining_axis != "other").sum()],
                                                                         [(r.defining_axis == "other").sum(), (r.defining_axis != "other").sum()]])[1]), 4),
             "axis_distribution_chi2_p": round(float(chi2_contingency(pd.crosstab(C.arm, C.defining_axis))[1]), 4),
             "note": "50 clusters per arm, one k-means seed, one labeller (Sonnet 5.5, claude -p); clusters, not prompts, are the units"}
json.dump(S, open(OUT / "summary.json", "w"), indent=1)
print({a: {k: S[a][k] for k in ["grab_bag_rule1_axis_other", "grab_bag_rule2_description_posthoc", "defining_axis"]} for a in ["far", "random"]}, S["test"])
