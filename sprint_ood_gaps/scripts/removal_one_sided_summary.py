"""Summary for scripts/removal_one_sided.py (designs/one_sided_negatives.md step 3): reads runs/removal_one_sided/queries.parquet,
adds Sohan's topic kind labels (exp80 harm, exp84 coding / non-English; 2 of 2,000 exp37 topics have no label and are dropped),
writes topics.parquet and summary.json."""
import json
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
ROOT = Path(__file__).resolve().parents[1]; OUT = ROOT / "runs/removal_one_sided"; TOP = 308
KINDS = {"safety": "is_harmful", "multilingual": "is_non_english", "coding": "is_coding"}
H = ROOT / "runs/removal_why/hf/results/raw"
harm = {json.loads(l)["id"]: json.loads(l)["is_harmful"] for l in open(H / "261028_exp80_safety_removal_v1/topics.jsonl")}
kl = {json.loads(l)["id"]: json.loads(l) for l in open(H / "261028_exp84_kind_labels_v1/topic_labels.jsonl")}
rows_out = pd.read_parquet(OUT / "queries.parquet")
tp = rows_out.groupby("topic").median(numeric_only=True).drop(columns="traffic_idx").reset_index()
tp = tp[("c" + tp.topic.astype(str)).isin(kl)].reset_index(drop=True); rows_out = rows_out[rows_out.topic.isin(tp.topic)]
cid = "c" + tp.topic.astype(str)
tp["is_harmful"] = cid.map(harm).fillna(False).astype(bool); tp["is_non_english"] = cid.map(lambda c: kl[c]["is_non_english"]).astype(bool)
tp["is_coding"] = cid.map(lambda c: kl[c]["is_coding"]).astype(bool); tp.to_parquet(OUT / "topics.parquet")

summ = {}
for kind, key in KINDS.items():
    y = tp[key].values; res = {"n_kind_topics": int(y.sum())}
    for det in ["k1", "mean5"]:
        def top_count(col): return int(y[np.argsort(-tp[col].values)[:TOP]].sum())
        base = top_count(f"full_{det}"); gain = top_count(f"{kind}_{det}") - base
        rgain = [top_count(f"random_{kind}_{s}_{det}") - base for s in range(3)]
        dk = tp[f"{kind}_{det}"] - tp[f"full_{det}"]; dr = [tp[f"random_{kind}_{s}_{det}"] - tp[f"full_{det}"] for s in range(3)]
        pq = rows_out.merge(tp[["topic", key]], on="topic")
        res[det] = {"kind_in_top_full": base, "gain": gain, "random_gains": rgain, "gain_minus_random_mean": round(gain - np.mean(rgain), 2),
                    "topic_auroc_change": round(roc_auc_score(y, dk), 3), "topic_auroc_change_random": [round(roc_auc_score(y, d), 3) for d in dr],
                    "prompt_auroc_change": round(roc_auc_score(pq[key], pq[f"{kind}_{det}"] - pq[f"full_{det}"]), 3),
                    "prompt_auroc_change_random": [round(roc_auc_score(pq[key], pq[f"random_{kind}_{s}_{det}"] - pq[f"full_{det}"]), 3) for s in range(3)],
                    "topic_auroc_level_after": round(roc_auc_score(y, tp[f"{kind}_{det}"]), 3), "topic_auroc_level_full": round(roc_auc_score(y, tp[f"full_{det}"]), 3)}
    summ[kind] = res
json.dump(summ, open(OUT / "summary.json", "w"), indent=1); print(json.dumps(summ, indent=1))
