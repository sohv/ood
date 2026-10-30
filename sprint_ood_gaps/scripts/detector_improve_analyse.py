"""Analysis for step 5 of designs/one_sided_negatives.md. For each split: candidates (a) rank-average MiniLM + tf-idf + activations
(layer picked on dev), (b) Mahalanobis PCA-256 per layer, (c) concatenated-layer kNN, (d) rank-average MiniLM + tf-idf; test AUROC
and paired bootstrap of (candidate - one-sided tf-idf), resampling groups for grouped splits, topics otherwise.
Selection on the primary arm (lsa) dev half. Writes runs/topic_split/improve_summary.json."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]; PRIMARY = "lsa"; rng = np.random.default_rng(0); B = 1000
summary = {}
for part in sys.argv[1:] or ["lsa", "minilm", "lsa_grouped", "minilm_grouped"]:
    D = ROOT / "runs/topic_split" / part; tp = pd.read_parquet(D / "topics.parquet")
    for f in ["candidates_b.parquet", "candidates_c.parquet"]:
        if (D / f).exists(): tp = tp.merge(pd.read_parquet(D / f), on="topic")
    pr = lambda c: tp[c].rank(pct=True)
    dev = tp.dev.values
    acts = [f"act_l{l}_knn_1" for l in (16, 20, 24)]
    best_act = max(acts, key=lambda c: roc_auc_score(tp.ood[dev], tp[c][dev]))
    tp["cand_a_rankavg_minilm_tfidf_act"] = (pr("v1_knn_kth_1") + pr("v1_tfidf") + pr(best_act)) / 3
    tp["cand_d_rankavg_minilm_tfidf"] = (pr("v1_knn_kth_1") + pr("v1_tfidf")) / 2
    cands = [c for c in tp.columns if c.startswith(("cand_", "maha_pca256", "knn1_concat"))]
    gfile = D / "groups.json"
    if part.endswith("_grouped") and not gfile.exists(): sys.exit(f"missing {gfile}: grouped partitions need it for the group bootstrap")
    unit = tp.topic.map({int(k): v for k, v in json.load(open(gfile)).items()}).values if gfile.exists() else tp.topic.values
    test = np.flatnonzero(~dev); y = tp.ood.values; uniq = np.unique(unit[test]); mem = {u: test[unit[test] == u] for u in uniq}
    boots = [np.concatenate([mem[u] for u in rng.choice(uniq, len(uniq))]) for _ in range(B)]
    res = {}
    for c in cands + ["v1_tfidf", "v1_knn_kth_1", best_act]:
        diffs = [roc_auc_score(y[b], tp[c].values[b]) - roc_auc_score(y[b], tp.v1_tfidf.values[b]) for b in boots]
        res[c] = {"dev_auroc": round(roc_auc_score(y[dev], tp[c][dev]), 3), "test_auroc": round(roc_auc_score(y[test], tp[c].values[test]), 3),
                  "test_minus_tfidf": round(roc_auc_score(y[test], tp[c].values[test]) - roc_auc_score(y[test], tp.v1_tfidf.values[test]), 3),
                  "diff_lo": round(np.percentile(diffs, 2.5), 3), "diff_hi": round(np.percentile(diffs, 97.5), 3)}
    summary[part] = {"bootstrap_unit": "group" if gfile.exists() else "topic", "activation_layer_for_a": best_act, "results": res,
                     "picked_on_dev": max(cands, key=lambda c: res[c]["dev_auroc"])}
    print(f"== {part} ({summary[part]['bootstrap_unit']} bootstrap); act layer for (a): {best_act}; picked on dev: {summary[part]['picked_on_dev']}")
    print(pd.DataFrame(res).T.to_string())
p = summary.get(PRIMARY)
if p:
    r = p["results"][p["picked_on_dev"]]
    p["verdict"] = "beats words" if r["diff_lo"] > 0 else "loses to words" if r["diff_hi"] < 0 else "tie"
    print("PRIMARY verdict:", p["picked_on_dev"], r, p["verdict"])
json.dump(summary, open(ROOT / "runs/topic_split/improve_summary.json", "w"), indent=1)
