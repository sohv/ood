"""Part 17 of designs/gap_taxonomy.md: three one-line rules (NAME_k placeholder, non-English, shared 60-character template prefix)
on the full one-sided loop's confirmed R50 topics vs its rejected R50 topics vs 56 random unconfirmed topics (seed 0). Members =
non-member traffic prompts (part 6 table). A topic is flagged by a rule when at least half its members match.
Writes runs/gap_taxonomy/heuristic/{topics.csv, summary.json}."""
import json, re
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import fisher_exact
ROOT = Path(__file__).resolve().parents[1]; OUT = ROOT / "runs/gap_taxonomy/heuristic"; OUT.mkdir(parents=True, exist_ok=True)
P = pd.read_parquet(ROOT / "runs/gap_taxonomy/blindspots/prompts.parquet")
texts = [json.loads(l)["text"] for l in open(ROOT / "runs/negatives/index/traffic_rows.jsonl")]
P["placeholder"] = [bool(re.search(r"NAME_\d", t)) for t in texts]
P["non_english"] = P["lang"] != "en"
P["template"] = P["template_prefix"]
M = P[~P.member & (P.topic >= 0)]
RULES = ["placeholder", "non_english", "template"]
T = M.groupby("topic")[RULES].mean(); T["n_members"] = M.groupby("topic").size()
F = (T[RULES] >= 0.5); F["any"] = F[RULES].any(axis=1)

V = ROOT / "runs/loop_onesided/v2"
conf = sorted({t["topic"] for g in json.load(open(V / "decision.json"))["gaps"]["R50"] for t in g})
rev = [json.loads(l) for l in open(V / "reviewed_topics_R50.jsonl")]
rej = sorted({int(r["topic"]) for r in rev} - set(conf))
sizes = P[P.topic >= 0].groupby("topic").size()
pool = sorted(int(t) for t in sizes[sizes >= 50].index if int(t) not in set(conf))
rnd = sorted(np.random.default_rng(0).choice(pool, 56, replace=False).tolist())
sets = {"confirmed": conf, "rejected": rej, "random": rnd}

rows, S = [], {"n_topics": {k: len(v) for k, v in sets.items()}, "rule": "topic flagged when >= half its non-member prompts match", "by_set": {}}
for name, ts in sets.items():
    f = F.reindex(ts).fillna(False)
    S["by_set"][name] = {**{r: int(f[r].sum()) for r in RULES + ["any"]},
                         "overlap": {"placeholder&template": int((f.placeholder & f.template).sum()), "placeholder&non_english": int((f.placeholder & f.non_english).sum()),
                                     "non_english&template": int((f.non_english & f.template).sum()), "all_three": int((f[RULES].all(axis=1)).sum())},
                         "share_any": round(float(f["any"].mean()), 3)}
    for t in ts: rows.append({"set": name, "topic": int(t), **{f"share_{r}": round(float(T.loc[t, r]), 3) if t in T.index else None for r in RULES},
                              **{f"flag_{r}": bool(F.loc[t, r]) if t in F.index else False for r in RULES + ["any"]}})
def fisher(a, b):
    x, y = S["by_set"][a]["any"], S["by_set"][b]["any"]; n, m = len(sets[a]), len(sets[b])
    o, p = fisher_exact([[x, n - x], [y, m - y]]); return {"odds_ratio": round(float(o), 2), "p_two_sided": f"{p:.3g}"}
S["fisher_any"] = {"confirmed_vs_random": fisher("confirmed", "random"), "confirmed_vs_rejected": fisher("confirmed", "rejected")}
pd.DataFrame(rows).to_csv(OUT / "topics.csv", index=False)
json.dump(S, open(OUT / "summary.json", "w"), indent=1); print(json.dumps(S, indent=1))
