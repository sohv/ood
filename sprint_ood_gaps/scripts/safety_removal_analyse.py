"""Safety-removal detectors, readouts (designs/safety_removal_detectors.md). AUROC positives vs matched controls per score and arm,
bootstrap over prompts (positives and controls resampled separately, same resample for every arm, 5,000); change from R0, and that change
minus Rc's. Share of positives above the gap line 12.1; median nearest-Dolci similarity of positives; nearest-source shares; jailbreak
split descriptive (10 positives). Outputs runs/safety_removal/{summary.csv, readouts.json}, figs/safety_removal/auroc_by_arm.png."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
ROOT = Path(__file__).resolve().parents[1]; OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "runs/safety_removal"
S = pd.read_parquet(OUT / "scores.parquet")
tf = OUT / "tfidf.parquet"
if tf.exists(): S = S.merge(pd.read_parquet(tf), on=["id", "arm"], how="left")
SCORES = [c for c in ("ratio", "distance", "tfidf") if c in S]; ARMS = ["R0", "R1", "R2", "R3", "Rc"]
W = {sc: S.pivot(index="id", columns="arm", values=sc)[ARMS] for sc in SCORES}
y = S.drop_duplicates("id").set_index("id").is_pos.loc[W[SCORES[0]].index].values
pi, ni = np.where(y == 1)[0], np.where(y == 0)[0]; rng = np.random.default_rng(0)
B = [np.r_[rng.choice(pi, len(pi)), rng.choice(ni, len(ni))] for _ in range(5000)]
rows = []
for sc in SCORES:
    X = W[sc].values; au = {a: roc_auc_score(y, X[:, j]) for j, a in enumerate(ARMS)}
    bt = np.array([[roc_auc_score(y[b], X[b, j]) for j in range(len(ARMS))] for b in B])
    for j, a in enumerate(ARMS):
        d = bt[:, j] - bt[:, 0]; dd = d - (bt[:, 4] - bt[:, 0])
        rows.append({"score": sc, "arm": a, "auroc": au[a], "lo": np.quantile(bt[:, j], .025), "hi": np.quantile(bt[:, j], .975),
                     "change_from_R0": au[a] - au["R0"], "change_lo": np.quantile(d, .025), "change_hi": np.quantile(d, .975),
                     "change_minus_Rc": (au[a] - au["R0"]) - (au["Rc"] - au["R0"]), "cmr_lo": np.quantile(dd, .025), "cmr_hi": np.quantile(dd, .975)})
T = pd.DataFrame(rows); T.to_csv(OUT / "summary.csv", index=False)
r3 = T[(T.arm == "R3")].set_index("score")
verdict = {sc: bool(r3.loc[sc, "change_from_R0"] >= 0.10 and r3.loc[sc, "change_lo"] > 0 and r3.loc[sc, "cmr_lo"] > 0) for sc in SCORES}
P = S[S.is_pos == 1]
R = {"n_pos": int(y.sum()), "n_ctrl": int((y == 0).sum()), "exact_dolci_member_share_pos": float(P.drop_duplicates("id").dolci_member.mean()),
     "share_pos_above_gap_line": {a: float((P[P.arm == a].ratio > 12.1413).mean()) for a in ARMS},
     "share_ctrl_above_gap_line": {a: float((S[(S.is_pos == 0) & (S.arm == a)].ratio > 12.1413).mean()) for a in ARMS},
     "median_nearest_sim_pos": {a: float(P[P.arm == a].nearest_sim.median()) for a in ARMS},
     "median_nearest_sim_ctrl": {a: float(S[(S.is_pos == 0) & (S.arm == a)].nearest_sim.median()) for a in ARMS},
     "pos_nearest_source_R0_top": P[P.arm == "R0"].nearest_source.value_counts().head(8).to_dict(),
     "ctrl_nearest_source_R0_top": S[(S.is_pos == 0) & (S.arm == "R0")].nearest_source.value_counts().head(5).to_dict(),
     "pos_nearest_is_safety_R0": float(P[P.arm == "R0"].nearest_source.isin(["WildGuardMix", "WildJailbreak", "CoCoNot"]).mean()),
     "jailbreak_split_R0_vs_R2_R1_median_distance": {g: {a: float(P[(P.arm == a) & (P.jailbreak_regex == jb)].distance.median()) for a in ("R0", "R1", "R2")}
                                                     for g, jb in (("jailbreak_regex_10", True), ("plain", False))},
     "verdict_detectors_see_missing_safety": verdict}
(OUT / "readouts.json").write_text(json.dumps(R, indent=1)); print(json.dumps(R, indent=1)); pd.set_option("display.width", 250); print(T.round(3).to_string(index=False))
fig, ax = plt.subplots(figsize=(7.5, 4)); x = np.arange(len(ARMS)); cols = {"ratio": "#3b6ea8", "distance": "#c77b30", "tfidf": "#5a9e5a"}
names = {"ratio": "gap score (k=50 ratio)", "distance": "distance (1 - nearest cosine)", "tfidf": "tf-idf traffic-vs-Dolci"}
for j, sc in enumerate(SCORES):
    q = T[T.score == sc].set_index("arm").loc[ARMS]; xx = x + (j - (len(SCORES) - 1) / 2) * 0.26
    ax.bar(xx, q.auroc, 0.26, color=cols[sc], label=names[sc]); ax.errorbar(xx, q.auroc, yerr=[q.auroc - q.lo, q.hi - q.auroc], fmt="none", ecolor="black", capsize=2, lw=.8)
ax.axhline(0.5, ls="--", c="0.4", lw=.8); ax.set_ylim(0.3, 1)
ax.set_xticks(x); ax.set_xticklabels(["R0\nnone removed", "R1\nWildGuardMix", "R2\nWildJailbreak", "R3\nall 3 safety", "Rc\nrandom, R3 size"], fontsize=8)
ax.set_ylabel("AUROC, harmful vs matched harmless traffic"); ax.legend(frameon=False, fontsize=7, loc="upper left")
ax.set_title("Detector AUROC for harmful traffic prompts after removing Dolci safety sources (Olmo-3 post-training data)", fontsize=9)
mod = OUT.name == "moderation"
sub_txt = (f"LMSYS only: {int(y.sum())} prompts flagged by OpenAI moderation vs {int((y==0).sum())} matched unflagged LMSYS prompts (length band)\nMiniLM; 95% bootstrap over prompts" if mod
           else f"{int(y.sum())} harmful (exp49_v2 grader) vs {int((y==0).sum())} matched harmless LMSYS/WildChat prompts; MiniLM; 95% bootstrap over prompts")
if mod: ax.set_ylabel("AUROC, flagged vs matched unflagged traffic"); ax.set_title("Detector AUROC, moderation-flagged traffic, after removing Dolci safety sources (Olmo-3 post-training data)", fontsize=8.5)
fig.text(0.5, 0.005, sub_txt, ha="center", fontsize=7, color="0.35")
fig.tight_layout(rect=(0, 0.07 if mod else 0.04, 1, 1)); fig.savefig(ROOT / ("figs/safety_removal/auroc_by_arm.png" if OUT.name == "safety_removal" else f"figs/safety_removal/auroc_by_arm_{OUT.name}.png"), dpi=150)
if "categories" in S and "pair" in S:
    base = S.drop_duplicates("id").set_index("id"); crow = []
    cats = base[base.is_pos == 1].categories.str.split(",").explode().value_counts()
    for c, n in cats.items():
        if n < 30: continue
        pp = base[(base.is_pos == 1) & base.categories.str.split(",").map(lambda x: c in x)].pair; ids = base[base.pair.isin(pp)].index
        for sc in SCORES:
            Xc = W[sc].loc[ids]; yc = base.loc[ids].is_pos.values; a0, a3, ac = (roc_auc_score(yc, Xc[a]) for a in ("R0", "R3", "Rc"))
            pi2, ni2 = np.where(yc == 1)[0], np.where(yc == 0)[0]; bb = [np.r_[rng.choice(pi2, len(pi2)), rng.choice(ni2, len(ni2))] for _ in range(1000)]
            dd = [(roc_auc_score(yc[b], Xc.R3.values[b]) - roc_auc_score(yc[b], Xc.R0.values[b])) - (roc_auc_score(yc[b], Xc.Rc.values[b]) - roc_auc_score(yc[b], Xc.R0.values[b])) for b in bb]
            crow.append({"category": c, "n_pos": int(n), "score": sc, "auroc_R0": a0, "auroc_R3": a3, "auroc_Rc": ac, "rise_minus_Rc": (a3 - a0) - (ac - a0), "lo": np.quantile(dd, .025), "hi": np.quantile(dd, .975)})
    C = pd.DataFrame(crow); C.to_csv(OUT / "by_category.csv", index=False); print(C.round(3).to_string(index=False))
