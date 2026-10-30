"""Register vs coverage test (designs/register_test.md). AUROC within Olmo-length bands (weighted by positive rows; bands
with >= 20 rows each side), higher = first set more gap-like. Scores: one_sided = 1 - nearest Dolci cosine, ratio,
tfidf_p_traffic, olmo_len. Writes runs/negatives/report/register_auroc.md and register_sets.csv."""
import glob, json
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
ROOT = Path(__file__).resolve().parents[1]
rows = []
for f in glob.glob(str(ROOT / "runs/negatives/scores/*.jsonl")) + glob.glob(str(ROOT / "runs/negatives/scores_register/*.jsonl")):
    for l in open(f):
        r = json.loads(l); rows.append({k: r.get(k) for k in ("set", "kind", "ratio", "nearest_mix_sim", "olmo_len", "tfidf_p_traffic", "exact_in_mix")})
d = pd.DataFrame(rows); d["one_sided"] = 1 - d["nearest_mix_sim"]
d = d[~((d.kind != "covered") & (d.exact_in_mix == True))].copy()
bands = [0, 16, 32, 64, 128, 256, 10**9]; d["band"] = pd.cut(d["olmo_len"], bands, right=False, labels=False)
S = ["one_sided", "ratio", "tfidf_p_traffic", "olmo_len"]
def strat(pos, neg, s):
    num = den = 0
    for b in range(len(bands) - 1):
        p, n = pos[pos.band == b], neg[neg.band == b]
        if len(p) >= 20 and len(n) >= 20:
            num += roc_auc_score(np.r_[np.ones(len(p)), np.zeros(len(n))], np.r_[p[s], n[s]]) * len(p); den += len(p)
    return num / den if den else np.nan
covered_reddit = sorted(d[d.kind == "register_covered"].set.unique())
negs = d[d.kind == "negative"]
comps = []
for a in ("pos_legal", "pos_medical", "reg_legaladvice2"):
    for b in covered_reddit:
        comps.append((a, b, d[d.set == a], d[d.set == b]))
    comps.append((a, "all covered-topic Reddit", d[d.set == a], d[d.kind == "register_covered"]))
for a in ("rw_legal", "rw_medical"):
    comps.append((a, "rw_wildchat", d[d.set == a], d[d.set == "rw_wildchat"]))
    comps.append((a, "23 negatives", d[d.set == a], negs))
comps.append(("all covered-topic Reddit", "23 negatives", d[d.kind == "register_covered"], negs))
out = [{"positive": a, "against": b, **{s: round(strat(p, n, s), 3) for s in S}} for a, b, p, n in comps]
t = pd.DataFrame(out); print(t.to_string(index=False))
med = d.groupby("set").agg(kind=("kind", "first"), n=("one_sided", "size"), median_one_sided=("one_sided", "median"), median_ratio=("ratio", "median"), median_len=("olmo_len", "median")).round(3)
sel = med[med.kind.isin(["register_covered", "register_uncovered", "rewrite_uncovered", "rewrite_control", "planted"])].sort_values("median_one_sided", ascending=False)
print(sel.to_string()); print("23 negatives pooled median one_sided:", round(negs.one_sided.median(), 3))
med.to_csv(ROOT / "runs/negatives/report/register_sets.csv")
(ROOT / "runs/negatives/report/register_auroc.md").write_text("## Register vs coverage, length-stratified AUROC\n\n" + t.to_markdown(index=False) + "\n\n## Medians\n\n" + sel.to_markdown() + f"\n\n23 negatives pooled median one_sided: {negs.one_sided.median():.3f}\n")
