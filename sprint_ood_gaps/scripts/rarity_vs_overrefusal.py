"""Does rarity in Dolci predict where Olmo over-refuses, beyond NAME_k and the toxic template? (2026-10-26, a check of Sohan's misbehaviour_predictor.)
Chain (written before running): rarity = pipeline_million gap score (ratio) and distance per prompt; outcome = over-refusal confirmed by both harm
readers (20k run, Sol + Astra via OpenRouter, every refused-harmless prompt labelled), because the grader-only label counts correct
refusals of explicit/jailbreak prompts, which are rare in Dolci; NAME_k and template prompts dropped individually; units = the 100k run's
750 k-means clusters (>= 20 remaining prompts); Spearman with a cluster bootstrap, plus prompt-level AUROC; bar 0.20 (Sohan's misbehaviour_predictor floor).
The grader-only outcome over all 100k is reported alongside to show the size of the explicit-content confound."""
import json, re
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
ROOT = Path(__file__).resolve().parents[1]; OUT = ROOT / "runs/rarity_check"; OUT.mkdir(parents=True, exist_ok=True)
P = {p["id"]: p for f in ["runs/misbehave/prompts.jsonl", "runs/misbehave100k/prompts.jsonl"] for p in map(json.loads, open(ROOT / f))}
cl = pd.read_json(ROOT / "runs/misbehave100k/stage1/rows.jsonl", lines=True)[["id", "cluster", "half", "graded", "refused_harmless", "name_k", "template"]]
g20 = pd.read_json(ROOT / "runs/misbehave/graded/graded.jsonl", lines=True)[["id", "over_refused"]]
d = cl.merge(g20, on="id", how="left"); d["ratio"] = d.id.map(lambda i: P[i]["ratio"]); d["distance"] = d.id.map(lambda i: P[i]["distance"])
d = d[d.graded & ~d.name_k.astype(bool) & ~d.template.astype(bool)].copy()
rng = np.random.default_rng(0); res = {}
def run(x, outcome, tag):
    c = x.groupby("cluster").agg(n=(outcome, "size"), rate=(outcome, "mean"), ratio=("ratio", "median"), distance=("distance", "median"))
    c = c[c.n >= 20]; r = {}
    for s in ["ratio", "distance"]:
        rho = spearmanr(c[s], c.rate)[0]; idx = np.arange(len(c))
        bs = [spearmanr(c[s].values[b], c.rate.values[b])[0] for b in (rng.choice(idx, len(idx)) for _ in range(2000))]
        auc = roc_auc_score(x[outcome].astype(int), x[s])
        q = pd.qcut(x[s].rank(method="first"), 5, labels=False); qr = x.groupby(q)[outcome].mean().round(4).tolist()
        r[s] = {"spearman_clusters": round(rho, 3), "ci95": [round(np.nanpercentile(bs, 2.5), 3), round(np.nanpercentile(bs, 97.5), 3)], "prompt_auroc": round(auc, 3), "rate_by_score_quintile": qr}
    res[tag] = {"clusters": int(len(c)), "prompts": int(len(x)), "positives": int(x[outcome].sum()), **r}
x20 = d[d.half == "20k"].copy(); x20["over_refused"] = x20.over_refused.fillna(False).astype(bool)
run(x20, "over_refused", "20k_confirmed_over_refusal (primary)")
d["refused_harmless"] = d.refused_harmless.astype(bool)
run(d, "refused_harmless", "100k_grader_refused_harmless (confounded by explicit/jailbreak)")
run(x20.assign(refused_harmless=x20.refused_harmless.astype(bool)), "refused_harmless", "20k_grader_refused_harmless (same prompts as primary)")
json.dump(res, open(OUT / "summary.json", "w"), indent=1); print(json.dumps(res, indent=1))
