"""Step 3b's measures, copied from scripts/removal_3b.py (block after "# measures") into a function so step 9 small scores every detector
the same way. Unchanged: queries kept only in topics with Gemini kind labels; safety/coding: topic AUROC of the change in topic-median score
(kind label vs rest) and the gain in kind topics among the top 308; multilingual arms: prompt AUROC of the change (py3langid non-English vs
English); each against the 20 random removals of the same size. Columns of q: "<det>__full" and "<det>__<arm>"."""
import json
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[3]
TOP, N_RAND = 308, 20
KINDS = ["safety", "coding", "aya_only", "all_non_english"]

def summarize(q, dets):
    H = ROOT / "runs/removal_why/hf/results/raw"
    harm = {json.loads(l)["id"]: json.loads(l)["is_harmful"] for l in open(H / "261028_exp80_safety_removal_v1/topics.jsonl")}
    kl = {json.loads(l)["id"]: json.loads(l) for l in open(H / "261028_exp84_kind_labels_v1/topic_labels.jsonl")}
    q = q[("c" + q.topic.astype(str)).isin(kl)]
    tlab = {"safety": lambda c: harm[c], "coding": lambda c: kl[c]["is_coding"]}
    summ = {}
    for det in dets:
        for k in KINDS:
            def prompt_auc(rn):
                y = q.non_en.values if k in ("aya_only", "all_non_english") else None
                if y is None: return None
                ch = q[f"{det}__{rn}"] - q[f"{det}__full"]; ok = ch.notna().values
                return float(roc_auc_score(y[ok], ch.values[ok]))
            def topic_stats(rn):
                if k not in tlab: return None
                tp = q.groupby("topic")[[f"{det}__full", f"{det}__{rn}"]].median()
                y = np.array([tlab[k](f"c{t}") for t in tp.index]).astype(bool)
                ch = (tp[f"{det}__{rn}"] - tp[f"{det}__full"]).values
                top = lambda col: int(y[np.argsort(-tp[col].values)[:TOP]].sum())
                return {"gain": top(f"{det}__{rn}") - top(f"{det}__full"), "auroc_change": float(roc_auc_score(y, ch))}
            real_p, real_t = prompt_auc(k), topic_stats(k)
            rp = [prompt_auc(f"rand_{k}_{s}") for s in range(N_RAND)]; rt = [topic_stats(f"rand_{k}_{s}") for s in range(N_RAND)]
            r = {}
            if real_p is not None:
                r["prompt_auroc_change"] = round(real_p, 3); r["random_prompt_auroc"] = [round(x, 3) for x in rp]
                r["rank_among_random"] = int(sum(x >= real_p for x in rp))
            if real_t is not None:
                r["topic_gain"] = real_t["gain"]; r["topic_auroc_change"] = round(real_t["auroc_change"], 3)
                r["random_gain"] = [x["gain"] for x in rt]; r["random_topic_auroc"] = [round(x["auroc_change"], 3) for x in rt]
                r["randoms_at_or_above_auroc"] = int(sum(x["auroc_change"] >= real_t["auroc_change"] for x in rt))
            summ[f"{det}/{k}"] = r
    return summ

def notices(r):
    """Step 9's reading, fixed before the run: AUROC of the change >= 0.70 and above all 20 randoms."""
    a = r.get("prompt_auroc_change", r.get("topic_auroc_change")); rnd = r.get("random_prompt_auroc", r.get("random_topic_auroc"))
    return bool(a >= 0.70 and all(a > x for x in rnd))
