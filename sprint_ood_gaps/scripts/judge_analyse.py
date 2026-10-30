"""Analyse the judge baseline (designs/judge_baseline.md; written before any judge output existed). Primary: AUROC of the
judge's 0-100 score against user real_gap on the topics the user labelled (unsure excluded), paired bootstrap of judge
minus distance. Secondary: same against Astra's real_gap; 3-way kappa with the user; real_gap share per arm. Unparsed
scores count as failed and the topic is dropped for that judge only (counted). Writes runs/judge/summary.json and the plot."""
import json
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, cohen_kappa_score
ROOT = Path(__file__).resolve().parents[1]; L = ROOT / "runs/rare_gaps/labels"; OUT = ROOT / "runs/judge"
key = json.load(open(L / "key.json")); user = {d["id"]: d["data"]["verdict"] for d in json.load(open(L / "labels_dump.json"))}
astra = json.load(open(L / "astra_labels.json")); tp = pd.read_csv(ROOT / "runs/rare_gaps/topics.csv").set_index("lab")
ORD = {"covered": 0, "partly": 1, "real_gap": 2}
base = pd.DataFrame([{"id": i, "arm": v["arm"], "user": user[i], "astra": astra[i], "distance": tp.loc[v["topic"]].median_score,
                      "ratio": tp.loc[v["topic"]].median_his, "astra_ord": ORD[astra[i]]} for i, v in key.items()]).set_index("id")
raw = pd.DataFrame([json.loads(l) for l in open(OUT / "raw_full.jsonl")])
judges = sorted({(m, a) for m, a in zip(raw.model, raw.arm)}, key=lambda x: ({"j50": 0, "hyb": 1, "same": 2}[x[1]], x[0]))
for m, a in judges:
    r = raw[(raw.model == m) & (raw.arm == a)].set_index("id"); c = f"{m.split('/')[1]}|{a}"
    base[c] = r.score; base[c + "_label"] = r.label
lab = base[base.user != "unsure"].copy(); lab["y"] = lab.user == "real_gap"; lab["y_astra"] = lab.astra == "real_gap"
cols = ["distance", "ratio", "astra_ord"] + [f"{m.split('/')[1]}|{a}" for m, a in judges]
rng = np.random.default_rng(0); B = [rng.integers(0, len(lab), len(lab)) for _ in range(4000)]
def auc(d, c, y): d = d[d[c].notna()]; return roc_auc_score(d[y], d[c])
res = {"n_labelled": len(lab), "n_real_gap": int(lab.y.sum()), "judges": {}}
for c in cols:
    ok = lab[c].notna(); d = lab[ok]
    bs, diff = [], []
    for b in B:
        s = lab.iloc[b]; s = s[s[c].notna()]
        if s.y.nunique() < 2: continue
        bs.append(roc_auc_score(s.y, s[c])); diff.append(bs[-1] - roc_auc_score(s.y, s.distance))
    e = {"n": int(ok.sum()), "failed": int((~ok).sum()), "auroc_user": round(auc(lab, c, "y"), 3), "ci": np.percentile(bs, [2.5, 97.5]).round(3).tolist(),
         "minus_distance": round(auc(d, c, "y") - roc_auc_score(d.y, d.distance), 3), "minus_distance_ci": np.percentile(diff, [2.5, 97.5]).round(3).tolist()}
    if c not in ("astra_ord",): e["auroc_astra"] = round(auc(lab, c, "y_astra"), 3)
    if c + "_label" in lab:
        d2 = lab[lab[c + "_label"].notna()]
        e["kappa3_user"] = round(cohen_kappa_score(d2.user, d2[c + "_label"]), 3); e["kappa3_astra"] = round(cohen_kappa_score(d2.astra, d2[c + "_label"]), 3)
        e["real_gap_share_by_arm"] = {a: f"{int((g[c + '_label'] == 'real_gap').sum())}/{g[c + '_label'].notna().sum()}" for a, g in base.groupby("arm")}
        e["score_label_spearman"] = round(float(d2[c].corr(d2[c + "_label"].map(ORD), method="spearman")), 3)
    res["judges"][c] = e
json.dump(res, open(OUT / "summary.json", "w"), indent=1); print(json.dumps(res, indent=1))
fig, ax = plt.subplots(figsize=(9.5, 4.4))
names = list(res["judges"]); v = [res["judges"][n]["auroc_user"] for n in names]; ci = np.array([res["judges"][n]["ci"] for n in names])
color = ["#777" if n in ("distance", "ratio") else "#a87600" if n == "astra_ord" else "#2a78d6" if n.endswith("j50") else "#7a5cc4" if n.endswith("hyb") else "#5ea2f0" for n in names]
ax.barh(range(len(names)), v, xerr=[np.array(v) - ci[:, 0], ci[:, 1] - np.array(v)], color=color, capsize=3)
ax.set_yticks(range(len(names)), [n.replace("astra_ord", "Astra labels (screens, cosine shown)").replace("|same", " | screen neighbour").replace("|j50", " | 50 tf-idf neighbours").replace("|hyb", " | MiniLM + tf-idf neighbours") for n in names])
ax.axvline(0.5, color="k", lw=0.8, ls=":"); ax.set_xlim(0, 1); ax.invert_yaxis(); ax.set_xlabel("AUROC against user real_gap label")
ax.set_title(f"Ranking user-labelled real gaps: gap scores vs LLM judges (Olmo-3-7B / Dolci topics)\n{len(lab)} topics labelled by one person ({int(lab.y.sum())} real gap), 3 arms of 30 picked by the scores; row bootstrap 95% CI", fontsize=9)
fig.tight_layout(); fig.savefig(ROOT / "figs/judge/auroc_vs_user_labels.png", dpi=150, bbox_inches="tight")
