"""Exploratory (2026-10-25, after the user asked whether "gap" is defined the wrong way): does a model-based gap score, the disagreement
between Olmo's two replies to the same prompt, predict poor answers better than the distance score?
Data already on disk: runs/behaviour_distance (2 replies per prompt, Olmo-3-7B-Instruct-DPO, T 0.6) and ood-03's strict Sole grades
(runs/judge/strict/raw_gpt-6.1-sol.jsonl, 2,700 prompts x 2 replies). Disagreement = 1 - cosine of the two replies (multilingual MiniLM).
Circularity: a prompt with one poor and one fine reply has dissimilar replies by construction, so the main test uses only prompts whose two
replies got the same grade (both poor vs both fine). CPU only."""
import json
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
ROOT = Path(__file__).resolve().parents[1]; OUT = ROOT / "runs/self_disagreement"; OUT.mkdir(parents=True, exist_ok=True)
g = pd.read_json(ROOT / "runs/judge/strict/raw_gpt-6.1-sol.jsonl", lines=True)
g["tdw"] = g.grade.map(lambda x: (x or {}).get("task_done_well")); g["r"] = g.id.str[-1].astype(int)
P = {p["id"]: p for p in map(json.loads, open(ROOT / "runs/behaviour_distance/prompts.jsonl"))}
R = {r["id"]: r["reply"] for r in map(json.loads, open(ROOT / "runs/behaviour_distance/out/rollouts_v1/rollouts.jsonl"))}
w = g.pivot(index="prompt_id", columns="r", values="tdw").rename(columns={0: "g0", 1: "g1"})
w = w[w.g0.isin(["yes", "no"]) & w.g1.isin(["yes", "no"])].copy()  # drop prompts with an n/a grade (refusals)
w["n_poor"] = (w.g0 == "no").astype(int) + (w.g1 == "no").astype(int)
from sentence_transformers import SentenceTransformer
m = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", device="cpu")
e0 = m.encode([R[f"{i}_r0"] for i in w.index], batch_size=64, normalize_embeddings=True, show_progress_bar=False)
e1 = m.encode([R[f"{i}_r1"] for i in w.index], batch_size=64, normalize_embeddings=True, show_progress_bar=False)
w["disagree"] = 1 - (e0 * e1).sum(1)
w["len"] = [np.log1p(len(R[f"{i}_r0"]) + len(R[f"{i}_r1"])) for i in w.index]
w["plen"] = [np.log1p(len(P[i]["text"])) for i in w.index]
z = np.load(ROOT / "runs/rare_gaps/one_sided.npz"); dist = 1 - z["top_s"].mean(1)
w["distance"] = [float(dist[P[i]["row"]]) if "row" in P[i] else np.nan for i in w.index]
w["arm"] = [P[i]["arm"] for i in w.index]; w["role"] = [P[i]["role"] for i in w.index]
w.to_csv(OUT / "per_prompt.csv")
res = {"n_prompts": int(len(w)), "n_poor_counts": w.n_poor.value_counts().sort_index().to_dict()}
same = w[w.n_poor != 1].copy(); same["both_poor"] = (same.n_poor == 2).astype(int)
def auc(df, col):
    df = df.dropna(subset=[col]); y = df.both_poor.values; s = df[col].values
    bs = [roc_auc_score(y[i], s[i]) for i in (np.random.default_rng(k).integers(0, len(df), len(df)) for k in range(500)) if 0 < y[i].mean() < 1]
    return {"n": int(len(df)), "auroc": round(roc_auc_score(y, s), 3), "ci95": [round(float(np.percentile(bs, 2.5)), 3), round(float(np.percentile(bs, 97.5)), 3)]}
res["same_grade_both_poor_vs_both_fine"] = {c: auc(same, c) for c in ["disagree", "distance", "len", "plen"]}
# does disagreement add to reply length + prompt length (logistic, 5-fold AUROC)
from sklearn.model_selection import cross_val_predict
X0 = same[["len", "plen"]].values; X1 = same[["len", "plen", "disagree"]].values; y = same.both_poor.values
res["cv_auroc_len_plen"] = round(roc_auc_score(y, cross_val_predict(LogisticRegression(max_iter=1000), X0, y, cv=5, method="predict_proba")[:, 1]), 3)
res["cv_auroc_len_plen_disagree"] = round(roc_auc_score(y, cross_val_predict(LogisticRegression(max_iter=1000), X1, y, cv=5, method="predict_proba")[:, 1]), 3)
res["spearman_all_disagree_vs_npoor"] = round(float(spearmanr(w.disagree, w.n_poor)[0]), 3)
res["spearman_disagree_vs_distance"] = round(float(spearmanr(w.disagree, w.distance, nan_policy="omit")[0]), 3)
q = pd.qcut(same.disagree, 5, labels=False); res["both_poor_rate_by_disagree_quintile"] = same.groupby(q).both_poor.mean().round(3).tolist()
q2 = pd.qcut(same.distance, 5, labels=False); res["both_poor_rate_by_distance_quintile"] = same.groupby(q2).both_poor.mean().round(3).tolist()
json.dump(res, open(OUT / "summary.json", "w"), indent=1); print(json.dumps(res, indent=1))
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(6.4, 4))
x = np.arange(1, 6)
ax.plot(x, res["both_poor_rate_by_disagree_quintile"], "o-", label="disagreement between Olmo's two replies")
ax.plot(x, res["both_poor_rate_by_distance_quintile"], "s--", label="distance to nearest Dolci prompts")
ax.set_xticks(x); ax.set_xlabel("score quintile (1 = lowest)"); ax.set_ylabel("share with both replies poor")
ax.set_title("Poor-answer rate by gap score, Olmo-3-7B-Instruct-DPO", fontsize=10)
fig.text(0.5, 0.005, f"{len(same)} prompts whose two replies got the same strict grade (Sol); behaviour_distance pool", ha="center", fontsize=8)
ax.legend(fontsize=8); fig.tight_layout(rect=(0, 0.03, 1, 1)); Path(ROOT / "figs/self_disagreement").mkdir(parents=True, exist_ok=True)
fig.savefig(ROOT / "figs/self_disagreement/poor_by_quintile.png", dpi=150)
