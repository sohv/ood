"""AUROC of each score for planted gaps (legal, medical) against negatives (unseen, same kind as Dolci sources), pooled
and per family, plus against covered Dolci rows. Higher = more gap-like for every score: ratio, ratio_fp32,
one-sided = 1 - nearest_mix_sim, olmo_len, tfidf_p_traffic. Bootstrap 95% CI over rows (1,000, seed 0) for the pooled
planted-vs-negatives comparison. Writes runs/negatives/report/auroc.csv and auroc.md."""
import glob, json
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
ROOT = Path(__file__).resolve().parents[1]
rows = []
for f in glob.glob(str(ROOT / "runs/negatives/scores/*.jsonl")):
    for l in open(f):
        r = json.loads(l); rows.append({k: r.get(k) for k in ("set", "family", "kind", "ratio", "ratio_fp32", "nearest_mix_sim", "olmo_len", "tfidf_p_traffic", "exact_in_mix", "tfidf_in_train")})
d = pd.DataFrame(rows); d["one_sided"] = 1 - d["nearest_mix_sim"]
n0 = len(d); d = d[~((d.kind == "negative") & (d.exact_in_mix == True))].copy(); print("dropped exact-member negatives:", n0 - len(d))
scores = ["ratio", "ratio_fp32", "one_sided", "olmo_len", "tfidf_p_traffic"]
rng = np.random.default_rng(0)
def auc(pos, neg, s, boot=False):
    y = np.r_[np.ones(len(pos)), np.zeros(len(neg))]; x = np.r_[pos[s].values, neg[s].values]
    a = roc_auc_score(y, x)
    if not boot: return a, None, None
    bs = []
    for _ in range(1000):
        i = rng.integers(0, len(pos), len(pos)); j = rng.integers(0, len(neg), len(neg))
        bs.append(roc_auc_score(np.r_[np.ones(len(i)), np.zeros(len(j))], np.r_[pos[s].values[i], neg[s].values[j]]))
    return a, np.percentile(bs, 2.5), np.percentile(bs, 97.5)
out = []
neg = d[d.kind == "negative"]; cov = d[d.kind == "covered"]
for p in ("pos_legal", "pos_medical"):
    pos = d[d.set == p]
    for s in scores:
        a, lo, hi = auc(pos, neg, s, boot=True)
        out.append({"planted": p, "against": "all negatives", "score": s, "auroc": a, "lo": lo, "hi": hi, "n_pos": len(pos), "n_neg": len(neg)})
        a, _, _ = auc(pos, cov, s)
        out.append({"planted": p, "against": "all covered", "score": s, "auroc": a, "lo": None, "hi": None, "n_pos": len(pos), "n_neg": len(cov)})
        for fam, g in neg.groupby("family"):
            a, _, _ = auc(pos, g, s)
            out.append({"planted": p, "against": f"neg:{fam}", "score": s, "auroc": a, "lo": None, "hi": None, "n_pos": len(pos), "n_neg": len(g)})
t = pd.DataFrame(out); t.to_csv(ROOT / "runs/negatives/report/auroc.csv", index=False)
piv = t[t.against.isin(["all negatives", "all covered"])].copy()
piv["cell"] = piv.apply(lambda r: f"{r.auroc:.3f}" + (f" [{r.lo:.3f}, {r.hi:.3f}]" if pd.notna(r.lo) else ""), axis=1)
md = piv.pivot_table(index=["planted", "against"], columns="score", values="cell", aggfunc="first")[scores]
fam = t[t.against.str.startswith("neg:")].pivot_table(index=["planted", "against"], columns="score", values="auroc")[scores].round(3)
(ROOT / "runs/negatives/report/auroc.md").write_text("## Planted vs negatives and covered, AUROC (higher = planted scored more gap-like)\n\n" + md.to_markdown() + "\n\n## Per negative family\n\n" + fam.to_markdown() + "\n")
print(md.to_string()); print(); print(fam.to_string())

# length-stratified AUROC: within Olmo token-length bands, weighted by planted rows per band (bands with >= 20 of each side)
bands = [0, 16, 32, 64, 128, 256, 10**9]
d["band"] = pd.cut(d["olmo_len"], bands, right=False, labels=False)
neg = d[d.kind == "negative"]
strat = []
for p in ("pos_legal", "pos_medical"):
    pos = d[d.set == p]
    for s in scores:
        num = den = 0; used = []
        for b in range(len(bands) - 1):
            pp, nn = pos[pos.band == b], neg[neg.band == b]
            if len(pp) >= 20 and len(nn) >= 20:
                num += auc(pp, nn, s)[0] * len(pp); den += len(pp); used.append((b, len(pp), len(nn)))
        strat.append({"planted": p, "score": s, "auroc_length_stratified": num / den if den else None, "planted_rows_used": den, "bands_used": used})
st = pd.DataFrame(strat); st.to_csv(ROOT / "runs/negatives/report/auroc_length_stratified.csv", index=False)
with open(ROOT / "runs/negatives/report/auroc.md", "a") as f:
    f.write("\n## Planted vs all negatives within Olmo-length bands (weighted by planted rows; bands with >= 20 rows each side)\n\n" + st.pivot(index="planted", columns="score", values="auroc_length_stratified")[scores].round(3).to_markdown() + "\n")
print(st.pivot(index="planted", columns="score", values="auroc_length_stratified")[scores].round(3).to_string())
print(st[["planted", "score", "planted_rows_used"]].drop_duplicates("planted").to_string(index=False))
