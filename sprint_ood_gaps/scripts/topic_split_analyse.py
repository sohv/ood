"""Analyse runs/topic_split/<partition>/ (designs/one_sided_negatives.md step 2). Usage: topic_split_analyse.py minilm [lsa]
Checks first (leaks), then topic-level AUROC OOD vs ID per detector (dev / test), precision in the top 100 / 300,
AUROC within topic-length bands, and V2 (adding ID halves to Dolci). Grouped splits (groups.json present) resample groups,
since ID/OOD was assigned per group; ungrouped splits resample topics. Writes runs/topic_split/summary.json."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]; IDX = ROOT / "runs/negatives/index"; rng = np.random.default_rng(0)
th = np.load(IDX / "traffic_hash.npy"); summary = {}
for part in sys.argv[1:]:
    D = ROOT / "runs/topic_split" / part; sp = json.load(open(D / "split.json"))
    split = pd.read_parquet(D / "split.parquet"); qs = pd.read_parquet(D / "queries.parquet"); tp = pd.read_parquet(D / "topics.parquet")
    id_topics = [int(t) for t, v in sp["ood"].items() if v == 0]
    ref = split[(split.half == "A") & split.topic.isin(id_topics)].traffic_idx.values
    # leak checks: no query text (hash) in the V1 reference; near-copy share by arm
    leak = int(np.isin(th[qs.traffic_idx.values], th[ref]).sum())
    near = qs.groupby("ood").v1_knn_kth_1.apply(lambda s: float((s <= 0.05).mean())).to_dict()
    out = {"topics": int(len(tp)), "ood_topics": int(tp.ood.sum()), "queries": int(len(qs)), "v1_reference": int(len(ref)),
           "query_hash_in_reference": leak, "share_queries_nn_cos_ge_0.95_by_ood": near}
    dets = [c for c in tp.columns if c.startswith(("v1_", "act_"))] + ["random"]
    tp["band"] = pd.qcut(tp.length.rank(method="first"), 4, labels=False)
    gfile = D / "groups.json"
    if part.endswith("_grouped") and not gfile.exists(): sys.exit(f"missing {gfile}: grouped partitions need it for the group bootstrap")
    unit = tp.topic.map({int(k): v for k, v in json.load(open(gfile)).items()}).values if gfile.exists() else tp.topic.values
    out["bootstrap_unit"] = "group" if gfile.exists() else "topic"
    def boot_idx(idx):
        # resample whole units (groups or topics) with replacement, keep all their topics
        u = unit[idx]; uniq = np.unique(u); members = {x: idx[u == x] for x in uniq}
        return [np.concatenate([members[x] for x in rng.choice(uniq, len(uniq))]) for _ in range(300)]
    res = {}
    for d in dets:
        r = {}
        for name, mask in [("dev", tp.dev.values), ("test", ~tp.dev.values), ("all", np.ones(len(tp), bool))]:
            idx = np.flatnonzero(mask); y, s = tp.ood.values, tp[d].values; a = roc_auc_score(y[idx], s[idx])
            bs = [roc_auc_score(y[b], s[b]) for b in boot_idx(idx) if 0 < y[b].sum() < len(b)]
            r[name] = {"auroc": round(a, 3), "lo": round(np.percentile(bs, 2.5), 3), "hi": round(np.percentile(bs, 97.5), 3)}
        top = tp.sort_values(d, ascending=False)
        r["precision_top100"] = round(top.ood.head(100).mean(), 3); r["precision_top300"] = round(top.ood.head(300).mean(), 3)
        r["auroc_within_length_bands"] = round(np.mean([roc_auc_score(g.ood, g[d]) for _, g in tp.groupby("band")]), 3)
        res[d] = r
    pick = max((d for d in dets if d != "random"), key=lambda d: res[d]["dev"]["auroc"])
    out.update({"detectors": res, "picked_on_dev": pick, "picked_test_auroc": res[pick]["test"]})
    if "v2_dolci_kth_1" in tp:
        v2 = {}
        for k in ["kth_1", "mean_5", "kth_20"]:
            a, b = tp[f"v2_dolci_{k}"], tp[f"v2_added_{k}"]; line = a.sort_values(ascending=False).iloc[299]
            v2[k] = {"line_natural_top300": round(float(line), 4),
                     "id_above_line_before": int((a[tp.ood == 0] >= line).sum()), "id_above_line_after": int((b[tp.ood == 0] >= line).sum()),
                     "ood_above_line_before": int((a[tp.ood == 1] >= line).sum()), "ood_above_line_after": int((b[tp.ood == 1] >= line).sum()),
                     "median_change_id": round(float((b - a)[tp.ood == 0].median()), 4), "median_change_ood": round(float((b - a)[tp.ood == 1].median()), 4),
                     "natural_auroc_ood_vs_id": round(roc_auc_score(tp.ood, a), 3)}
        out["v2"] = v2
    summary[part] = out
    print(f"== {part}: {out['topics']} topics, {out['queries']} queries, leak {leak}, near-copy share {near}")
    print(pd.DataFrame({d: {"dev": r["dev"]["auroc"], "test": r["test"]["auroc"], "p@100": r["precision_top100"], "p@300": r["precision_top300"],
                            "in-band": r["auroc_within_length_bands"]} for d, r in res.items()}).T.to_string())
    print("picked on dev:", pick, res[pick]["test"])
    if "v2" in out: print(pd.DataFrame(out["v2"]).T.to_string())
json.dump(summary, open(ROOT / "runs/topic_split/summary.json", "w"), indent=1)
