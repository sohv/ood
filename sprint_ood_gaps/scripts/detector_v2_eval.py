"""Evaluate frozen detector v2 (designs/detector_v2.md) on the 42 judge_best labels (primary) and the 81 rare_gaps labels
(secondary). Written at freeze time, before any judge_best label existed. Usage: detector_v2_eval.py labels_dump.json
(the judge_best page's 'labels' collection exported as [{id, data: {verdict}}])."""
import hashlib, json, sys
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
ROOT = Path(__file__).resolve().parents[1]; OUT = ROOT / "runs/detector_v2"
for line in [l for l in (OUT / "FROZEN.txt").read_text().splitlines()[1:] if not l.startswith("#")]:
    h, name = line.split("  "); f = ROOT / name if "/" in name else OUT / name; assert hashlib.sha256(f.read_bytes()).hexdigest() == h, f"{name} changed after freeze"
d = pd.read_csv(OUT / "scores.csv").set_index("topic"); top30 = json.load(open(OUT / "top30.json"))
COLS = ["b_c", "two_stage", "a", "b", "v1", "judge"]
def evaluate(y):  # y: Series topic -> bool real_gap, can't-tell already dropped
    x = d.loc[y.index].assign(y=y); rng = np.random.default_rng(0); B = [rng.integers(0, len(x), len(x)) for _ in range(4000)]
    res = {"n": len(x), "real_gap": int(x.y.sum())}
    for c in COLS:
        a = roc_auc_score(x.y, x[c]); dv, dj = [], []
        for b in B:
            s = x.iloc[b]
            if s.y.nunique() < 2: continue
            v = roc_auc_score(s.y, s[c]); dv.append(v - roc_auc_score(s.y, s.v1)); dj.append(v - roc_auc_score(s.y, s.judge))
        lab30 = [t for t in top30.get(c, []) if t in y_all.index]
        res[c] = {"auroc": round(a, 3), "minus_v1_ci": np.percentile(dv, [2.5, 97.5]).round(3).tolist(), "minus_judge_ci": np.percentile(dj, [2.5, 97.5]).round(3).tolist(),
                  "top30_labelled": len(lab30), "top30_real_gap": int(y_all.loc[lab30].sum()) if lab30 else None}
    p = res["b_c"]; res["b_c_better_than_v1"] = p["auroc"] >= res["v1"]["auroc"] + 0.10 and p["minus_v1_ci"][0] > 0
    res["b_c_matches_judge"] = p["auroc"] >= res["judge"]["auroc"] - 0.05
    return res
new = {r["id"]: r["data"]["verdict"] for r in json.load(open(sys.argv[1]))}; key = json.load(open(ROOT / "runs/judge_best/labels/key.json"))
y_all = pd.Series({key[i]["topic"]: v == "real_gap" for i, v in new.items() if v != "unsure"})  # all labelled topics: top-30 precision
y_new = pd.Series({key[i]["topic"]: v == "real_gap" for i, v in new.items() if v != "unsure" and key[i]["arm"] in ("judge_only", "v2_only")})  # the 39 labelled after the trim: AUROC
L = ROOT / "runs/rare_gaps/labels"; old = {r["id"]: r["data"]["verdict"] for r in json.load(open(L / "labels_dump.json"))}; ok = json.load(open(L / "key.json"))
y_old = pd.Series({ok[i]["topic"]: v == "real_gap" for i, v in old.items() if v != "unsure"})
out = {"primary_39": evaluate(y_new), "secondary_81_auroc_only": evaluate(y_old), "labelled_new": len(new), "unsure_new": sum(v == "unsure" for v in new.values())}
json.dump(out, open(OUT / "eval.json", "w"), indent=1); print(json.dumps(out, indent=1))
