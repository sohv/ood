"""Trigger scan, behaviour screen (exploratory, added after the Dolci scan): for each pattern with Dolci lower-bound lift >= 3
and >= 100 traffic prompts, Olmo's refusal rate on harmless prompts that contain it vs those that do not, from the behaviour
check (runs/behaviour_distance, Sohan's exp49_v2 grader; 2 replies per prompt, prompt = unit). Output runs/triggers/behaviour_screen.csv."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
from statsmodels.stats.proportion import proportion_confint
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "scripts"))
from trig_scan_feats import feats
P = pd.read_csv(ROOT / "runs/triggers/patterns.csv"); P = P[P.traffic_n >= 100]
pr = {r["id"]: r["text"] for r in map(json.loads, open(ROOT / "runs/behaviour_distance/prompts.jsonl"))}
g = pd.read_json(ROOT / "runs/behaviour_distance/graded/graded.jsonl", lines=True); g = g[g.graded]
u = g.groupby("prompt_id").agg(harmful=("harmful_req", "max"), refused=("refused", "mean")).reset_index()
u = u[u.harmful.astype(str).str.lower().isin(["false", "0", "0.0"])]; u["f"] = u.prompt_id.map(lambda i: feats(pr[i][:3000]))
rows = []
for k in P.pattern:
    m = u.f.map(lambda s: k in s).values; n = int(m.sum())
    if n < 15: continue
    a, b = u.refused[m].mean(), u.refused[~m].mean()
    lo, hi = proportion_confint(int(round(u.refused[m].sum())), n, method="wilson")
    rows.append({"pattern": k, "harmless_prompts_with": n, "refusal_with": a, "refusal_with_lo": lo, "refusal_with_hi": hi, "refusal_without": b, "diff": a - b})
B = pd.DataFrame(rows).merge(P[["pattern", "dolci_n", "rate", "lift_lo", "top_refused_source", "traffic_n"]], on="pattern").sort_values("refusal_with_lo", ascending=False)
B.to_csv(ROOT / "runs/triggers/behaviour_screen.csv", index=False)
print("harmless prompts", len(u), "base refusal", round(u.refused.mean(), 3), "patterns screened", len(B))
pd.set_option("display.width", 250); print(B.head(50).round(3).to_string(index=False))
