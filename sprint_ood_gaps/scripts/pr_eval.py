"""Precision-recall with negatives and positives in one pool (Stefan 28 Oct: "you have to have a test set, a negative set ...
any method that outputs high scores will get a high score"). Uses existing scores only; no new runs.
A. Topic split (runs/topic_split/<part>/topics.parquet): positives = topics whose A half was discarded (OOD), negatives = topics whose
   A half was added to the reference (ID). Per detector: average precision, precision at K = 50/100/300, recall at those K, and the
   precision of a detector that ranks at random (= share of positives), test half.
B. Removal test (runs/removal_3b/queries.parquet): rank topics by the score change after removal; positives = topics of the removed
   kind (Gemini labels), negatives = all other topics. Average precision and precision at K for the kind removal and each of 20 randoms.
Writes runs/pr_eval/summary.json."""
import json
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import average_precision_score

ROOT = Path(__file__).resolve().parents[1]; OUT = ROOT / "runs/pr_eval"; OUT.mkdir(exist_ok=True)
def at_k(y, s, ks=(50, 100, 300)):
    o = np.argsort(-s); return {f"precision@{k}": round(float(y[o[:k]].mean()), 3) for k in ks} | {f"recall@{k}": round(float(y[o[:k]].sum() / y.sum()), 3) for k in ks}
summ = {"A_topic_split": {}, "B_removal": {}}
for part in ["minilm", "lsa", "minilm_grouped", "lsa_grouped"]:
    tp = pd.read_parquet(ROOT / f"runs/topic_split/{part}/topics.parquet"); t = tp[~tp.dev.values]
    y = t.ood.values.astype(int); dets = [c for c in t.columns if c.startswith(("v1_", "act_")) and "maha" not in c] + ["random"]
    summ["A_topic_split"][part] = {"n_topics_test": int(len(t)), "positives": int(y.sum()), "base_precision": round(float(y.mean()), 3),
                                   **{d: {"average_precision": round(float(average_precision_score(y, t[d].values)), 3), **at_k(y, t[d].values)} for d in dets}}
q = pd.read_parquet(ROOT / "runs/removal_3b/queries.parquet")
H = ROOT / "runs/removal_why/hf/results/raw"
harm = {int(json.loads(l)["id"][1:]): json.loads(l)["is_harmful"] for l in open(H / "261028_safety_removal_v1/topics.jsonl")}
kl = {int(json.loads(l)["id"][1:]): json.loads(l) for l in open(H / "261028_removal_code_multilingual_kind_labels_v1/topic_labels.jsonl")}
q = q[q.topic.isin(kl)]
for kind, lab in [("safety", lambda t: harm[t]), ("coding", lambda t: kl[t]["is_coding"])]:
    for det in ["minilm", "tfidf"]:
        tp = q.groupby("topic")[[c for c in q.columns if c.startswith(f"{det}__")]].median()
        y = np.array([lab(t) for t in tp.index]).astype(int)
        def pr(arm):
            ch = (tp[f"{det}__{arm}"] - tp[f"{det}__full"]).values; return {"average_precision": round(float(average_precision_score(y, ch)), 3), **at_k(y, ch, (20, 50, 100))}
        real = pr(kind); rnd = [pr(f"rand_{kind}_{s}") for s in range(20)]
        summ["B_removal"][f"{det}/{kind}"] = {"positives": int(y.sum()), "topics": int(len(y)), "base_precision": round(float(y.mean()), 3), "kind_removal": real,
            "random_average_precision_max": max(r["average_precision"] for r in rnd), "random_average_precision_median": float(np.median([r["average_precision"] for r in rnd])),
            "randoms_at_or_above_kind_AP": int(sum(r["average_precision"] >= real["average_precision"] for r in rnd))}
json.dump(summ, open(OUT / "summary.json", "w"), indent=1)
for part, v in summ["A_topic_split"].items():
    print(part, "base", v["base_precision"], {d: (v[d]["average_precision"], v[d]["precision@100"]) for d in v if isinstance(v[d], dict)})
for k, v in summ["B_removal"].items(): print(k, "base", v["base_precision"], v["kind_removal"], "random AP max", v["random_average_precision_max"])
