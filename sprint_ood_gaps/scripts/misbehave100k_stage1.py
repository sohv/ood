"""Stage 1 of the two-stage 100k analysis (2026-10-26, before reading any 80k over-refusal result): cluster enrichment of
grader refused-harmless over all 100,000 prompts, no harm re-read. Stage 2 re-reads only prompts in the clusters that pass here
(plus a random 400 for the base benign share). Same clustering as misbehave100k_analyse.py: spherical k-means, 750 clusters, seed 0."""
import json, re, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import fisher_exact
ROOT = Path(__file__).resolve().parents[1]; OUT = ROOT / "runs/misbehave100k/stage1"; OUT.mkdir(parents=True, exist_ok=True)
text = {p["id"]: p["text"] for f in ["runs/misbehave/prompts.jsonl", "runs/misbehave100k/prompts.jsonl"] for p in map(json.loads, open(ROOT / f))}
row_of = {p["id"]: p["row"] for f in ["runs/misbehave/prompts.jsonl", "runs/misbehave100k/prompts.jsonl"] for p in map(json.loads, open(ROOT / f))}
old = pd.read_json(ROOT / "runs/misbehave/graded/graded.jsonl", lines=True); old["half"] = "20k"
new = pd.concat([pd.read_json(ROOT / f, lines=True) for f in ["runs/misbehave100k/pilot_graded/graded.jsonl"] + [f"runs/misbehave100k/graded_shard{k}/graded.jsonl" for k in range(4)]]); new["half"] = "80k"
keep = ["id", "half", "name_k", "graded", "refused", "refused_harmless"]
d = pd.concat([old[keep], new[keep]], ignore_index=True); assert d.id.is_unique and len(d) == 100000, len(d)
for c in ["refused", "refused_harmless"]: d[c] = d[c].fillna(False).astype(bool)
d["template"] = d.id.map(text).str.contains(r"say something (?:toxic|bad|harmful|benign)[^\n]{0,20}\[your answer\]", case=False, regex=True)
E = np.load(ROOT / "runs/negatives/index/traffic.npy", mmap_mode="r")[[row_of[i] for i in d.id]].astype(np.float32); E /= np.linalg.norm(E, axis=1, keepdims=True)
from sklearn.cluster import KMeans
d["cluster"] = KMeans(750, n_init=2, random_state=0).fit_predict(E)
d.to_json(OUT / "rows.jsonl", orient="records", lines=True)
g = d[d.graded]; K0 = int(g.refused_harmless.sum()); N = len(g); rows = []
for c, x in g.groupby("cluster"):
    if len(x) < 20: continue
    k, n = int(x.refused_harmless.sum()), len(x)
    rows.append({"cluster": int(c), "n": n, "refused_harmless": k, "rate": round(k / n, 3), "n_80k": int((x.half == "80k").sum()), "rh_80k": int(x[x.half == "80k"].refused_harmless.sum()),
                 "name_k_share": round(float(x.name_k.mean()), 2), "template_share": round(float(x.template.mean()), 2),
                 "p": fisher_exact([[k, n - k], [K0 - k, N - n - (K0 - k)]], alternative="greater")[1]})
t = pd.DataFrame(rows).sort_values("p"); m = len(t); t["p_holm"] = np.minimum(1, (t.p * (m - np.arange(m))).cummax()); t.to_csv(OUT / "clusters.csv", index=False)
cand = t[(t.p_holm < 0.05) & (t.refused_harmless >= 5)]
print("graded", N, "refused_harmless", K0, round(K0 / N, 4), "clusters tested", m, "candidates", len(cand))
print(cand.drop(columns=["p"]).to_string(index=False))
