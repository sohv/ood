"""Analyse step 9 (revised): Olmo activation kNN on step 3b's removal test, with step 3b's measures, labels and randoms;
plus part B (family hold-out at full scale). Inputs runs/step9/out/{gates.json, scores_L*.npz, partb_L*.npz}.
Writes runs/step9/summary.json."""
import glob, json
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]; D = ROOT / "runs/step9/out"; TOP = 308
q = pd.read_parquet(ROOT / "runs/removal_3b/queries.parquet")[["traffic_idx", "topic", "non_en"]].reset_index(drop=True)
H = ROOT / "runs/removal_why/hf/results/raw"
harm = {json.loads(l)["id"]: json.loads(l)["is_harmful"] for l in open(H / "261028_exp80_safety_removal_v1/topics.jsonl")}
kl = {json.loads(l)["id"]: json.loads(l) for l in open(H / "261028_exp84_kind_labels_v1/topic_labels.jsonl")}
tlab = {"safety": lambda c: harm[c], "coding": lambda c: kl[c]["is_coding"]}
labelled = ("c" + q.topic.astype(str)).isin(kl).values
summ = {"gates": json.load(open(D / "gates.json"))}
for f in sorted(glob.glob(str(D / "scores_L*.npz"))):
    L = Path(f).stem.split("_L")[1]; z = np.load(f)
    for mode, suf in [("own_std", ""), ("fixed_std", "_fixedstd")]:
        for det in ["k1", "m5"]:
            full = z[f"full__{det}"]
            for k in ["safety", "coding", "aya_only", "all_non_english"]:
                def change(rn): return z[f"{rn}__{det}{suf}"] - full
                r = {}
                if k in tlab:
                    def topic(rn):
                        tp = pd.DataFrame({"topic": q.topic, "b": full, "a": z[f"{rn}__{det}{suf}"]})[labelled].groupby("topic").median()
                        y = np.array([tlab[k](f"c{t}") for t in tp.index]).astype(bool)
                        top = lambda col: int(y[np.argsort(-tp[col].values)[:TOP]].sum())
                        return top("a") - top("b"), float(roc_auc_score(y, (tp.a - tp.b).values))
                    g, a = topic(k); rs = [topic(f"rand_{k}_{s}") for s in range(20)]
                    r.update(topic_gain=g, topic_auroc_change=round(a, 3), random_gain=[x[0] for x in rs], random_topic_auroc=[round(x[1], 3) for x in rs],
                             randoms_at_or_above=int(sum(x[1] >= a for x in rs)))
                else:
                    # same rows as step 3b: queries from the 1,998 topics that have kind labels
                    y = q.non_en.values[labelled]
                    a = float(roc_auc_score(y, change(k)[labelled])); rs = [float(roc_auc_score(y, change(f"rand_{k}_{s}")[labelled])) for s in range(20)]
                    r.update(prompt_auroc_change=round(a, 3), random_prompt_auroc=[round(x, 3) for x in rs], randoms_at_or_above=int(sum(x >= a for x in rs)))
                summ[f"L{L}/{mode}/{det}/{k}"] = r
# part B: family hold-out, AUROC per family, dev/test split as step 1; also with identical-vector rows dropped on both sides (MiniLM nn1 >= 0.999, audit O11-2)
split = json.load(open(ROOT / "runs/one_sided/families.json"))["split"]
for f in sorted(glob.glob(str(D / "partb_L*.npz"))):
    L = Path(f).stem.split("_L")[1]; z = np.load(f)
    for det in ["k1", "m5"]:
        per = {}
        for fam in split:
            rows = pd.read_parquet(ROOT / f"runs/one_sided/rows_{fam}.parquet"); s = z[f"{fam}__{det}"]
            keep = rows.nn1_sim_after.values < 0.999
            per[fam] = {"split": split[fam], "auroc": round(float(roc_auc_score(rows.label, s)), 3),
                        "auroc_no_identical": round(float(roc_auc_score(rows.label[keep], s[keep])), 3)}
        for sp in ["dev", "test"]:
            per[f"mean_{sp}"] = round(float(np.mean([v["auroc"] for k2, v in per.items() if isinstance(v, dict) and v.get("split") == sp])), 3)
        summ[f"partB/L{L}/{det}"] = per
json.dump(summ, open(ROOT / "runs/step9/summary.json", "w"), indent=1)
for k, v in summ.items():
    if k.startswith("L") and "/own_std/" in k:
        print(k, {kk: vv for kk, vv in v.items() if not kk.startswith("random_")})
