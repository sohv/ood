"""Sohan's misbehaviour_predictor (gap-score rank vs per-topic refusal effect) without NAME_k / toxic-template topics, 2026-10-26.
NAME_k and template share per topic is measured on the 5 member prompts exp52 actually tested (detectood/ood
data/processed/gap_overrefusal/prompts.jsonl), since the refusal effect is measured on exactly those prompts. A topic is excluded
if >= half its tested prompts carry NAME_k or the template. Spearman with a topic bootstrap (1,000, seed 0).
Usage: python scripts/misbehaviour_predictor_tested_prompt_check.py <gap_overrefusal prompts.jsonl>"""
import json, re, sys, collections, numpy as np, pandas as pd
from scipy.stats import spearmanr
rows = [json.loads(l) for l in open(sys.argv[1])]
nk = re.compile(r"NAME_\d"); tp = re.compile(r"[Ss]ay something (toxic|bad|harmful|benign)")
m = collections.defaultdict(lambda: [0, 0, 0])
for r in rows:
    if r["role"] != "member": continue
    c = m[r["group"].split("_")[1]]; c[0] += 1; c[1] += bool(nk.search(r["text"])); c[2] += bool(tp.search(r["text"]))
T = pd.read_json("runs/sohan_misbehaviour_predictor/results/raw/261026_misbehaviour_predictor_behaviour_predictors_v1/topics.jsonl", lines=True)
T["nk"] = T.id.map(lambda i: m[i][1] / m[i][0]); T["tp"] = T.id.map(lambda i: m[i][2] / m[i][0])
def rho(x):
    r = spearmanr(x.score_neighbour_count, x.refusal_effect)[0]; rng = np.random.default_rng(0)
    b = [spearmanr(*x.sample(len(x), replace=True, random_state=int(s))[["score_neighbour_count", "refusal_effect"]].T.values)[0] for s in rng.integers(0, 1e9, 1000)]
    return {"rho": round(r, 3), "ci95": [round(np.percentile(b, 2.5), 3), round(np.percentile(b, 97.5), 3)], "n": len(x)}
art = (T.nk >= 0.5) | (T.tp >= 0.5); c = T[~art]
out = {"excluded": int(art.sum()), "all_221": rho(T), "kept_all": rho(c), "kept_gap": rho(c[c.arm == "gap"]), "kept_random": rho(c[c.arm == "random"])}
json.dump(out, open("runs/rarity_check/misbehaviour_predictor_tested_prompt_check.json", "w"), indent=1); print(json.dumps(out, indent=1))
