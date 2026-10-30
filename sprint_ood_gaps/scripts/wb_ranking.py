# ranking_variants planted-needle ranking bench with word baselines. Pool, needle planting, top-100 metrics, pct20 length control,
# jitter and whole-cluster review follow scripts/run_ranking_bench.py:139-196, 214-275 and src/metrics/ranking.py:26-40.
# Background: density_ratio_needles WildChat 10,000 (all languages) + iterative_discovery LMSYS 10,000; per needle seed s, rows s*100..s*100+99 of each
# 300-row bank (legal, medical = gap; aya, magicoder = covered control). Mix = repro_two_sided idfit + idmix (4,000); Sohan's
# tf-idf arm keeps his idfit-only fit (run_ranking_bench.py:177-179).
# usage: .venv/bin/python -m scripts.wb_ranking [--limit 300 --tag tiny]

import argparse
import csv
import json
import re
import time

import numpy as np
from scipy.sparse import vstack
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from scripts.wb_common import (INPUTS, RUN, TYPES, cluster_order, jitter_order, length_controlled, load_jsonl,
                               top_metrics, write_json, write_jsonl)
from scripts.wb_detectors import ALL_ARMS, ONE_SIDED, compute_arms

NEEDLE_SEEDS = (0, 1, 2)
N_PER_TYPE = 100
TOP_K = 100
KMEANS_K = 200


def cross_fit_tfidf(deploy, mix, c, folds, seed):
    # src/metrics/density_ratio.py:46-56 (deploy-vs-mix density ratio, density_ratio_needles/07 settings C = 1.0, 5 folds)
    vec = TfidfVectorizer(sublinear_tf=True, min_df=2, max_features=200_000).fit(deploy + mix)
    xd, xm = vec.transform(deploy), vec.transform(mix)
    fold = np.random.default_rng(seed).permutation(len(deploy)) % folds
    scores = np.zeros(len(deploy))
    for f in range(folds):
        x = vstack([xd[fold != f], xm])
        y = np.r_[np.ones(int((fold != f).sum())), np.zeros(len(mix))]
        clf = LogisticRegression(C=c, class_weight="balanced", max_iter=2000, random_state=seed).fit(x, y)
        scores[fold == f] = clf.decision_function(xd[fold == f])
    return scores


def kmeans_labels(features, k, seed, n_init=3):
    # src/metrics/ranking.py:26-28
    z = (features - features.mean(0)) / (features.std(0) + 1e-6)
    return KMeans(n_clusters=k, n_init=n_init, random_state=seed).fit_predict(z)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="background rows per source (tiny run)")
    ap.add_argument("--tag", default="full")
    args = ap.parse_args()
    t0 = time.time()
    lim = args.limit
    lens_all = {}
    for p in ("deploy", "lmsys", *[f"needles_{t}" for t in TYPES]):
        lens_all |= json.loads((RUN / "cache" / "lengths" / "ranking_variants" / f"{p}.json").read_text())["lengths"]

    def emb_of(e, pool, ids):
        f = np.load(RUN / "cache" / f"emb_{args.tag}" / e / f"{pool.replace('/', '__')}.npz")
        assert list(f["ids"][: len(ids)]) == ids, (e, pool)
        return f["emb"][: len(ids)]

    srcs = {"wildchat": "ranking_variants/density_ratio_needles/deploy", "lmsys": "ranking_variants/iterative_discovery/lmsys",
            **{t: f"ranking_variants/density_ratio_needles/needles_{t}" for t in TYPES}}
    data = {}
    for name, pool in srcs.items():
        rows = load_jsonl(INPUTS / f"{pool}.jsonl")
        rows = rows[:lim] if (lim and name in ("wildchat", "lmsys")) else rows
        if lim and name in TYPES:
            rows = rows[: 3 * min(N_PER_TYPE, lim // 10 or 1)]
        data[name] = rows
    n_per = N_PER_TYPE if not lim else min(N_PER_TYPE, lim // 10 or 1)
    bg = data["wildchat"] + data["lmsys"]
    fixed_pools = {"deploy": bg, **{t: data[t] for t in TYPES}}
    mix_rows = load_jsonl(INPUTS / "repro_two_sided/idfit.jsonl") + load_jsonl(INPUTS / "repro_two_sided/idmix.jsonl")
    idfit_rows = load_jsonl(INPUTS / "repro_two_sided/idfit.jsonl")
    dolma_rows = load_jsonl(INPUTS / "repro_two_sided/dolma.jsonl")
    if lim:
        idfit_rows, dolma_rows = idfit_rows[:lim], dolma_rows[:lim]
        mix_rows = idfit_rows + load_jsonl(INPUTS / "repro_two_sided/idmix.jsonl")[:lim]
    mix = [r["text"] for r in mix_rows]
    ref = [" ".join(r["text"].split()[:400]) for r in dolma_rows]
    tests = {p: [r["text"] for r in rows] for p, rows in fixed_pools.items()}
    lengths = {p: np.array([lens_all[r["id"]] for r in rows]) for p, rows in fixed_pools.items()}

    emb = {}
    for e in ("minilm", "mminilm"):
        n_fit = len(idfit_rows) if not lim else lim
        fit_ids = [r["id"] for r in load_jsonl(INPUTS / "repro_two_sided/idfit.jsonl")][: (n_fit if lim else None)]
        mix_e = np.vstack([emb_of(e, "repro_two_sided/idfit", fit_ids),
                           emb_of(e, "repro_two_sided/idmix", [r["id"] for r in load_jsonl(INPUTS / "repro_two_sided/idmix.jsonl")][: (lim if lim else None)])])
        emb[e] = {"mix": mix_e, "ref": emb_of(e, "repro_two_sided/dolma", [r["id"] for r in dolma_rows])}
        emb[e]["deploy"] = np.vstack([emb_of(e, srcs["wildchat"], [r["id"] for r in data["wildchat"]]),
                                      emb_of(e, srcs["lmsys"], [r["id"] for r in data["lmsys"]])])
        for t in TYPES:
            emb[e][t] = emb_of(e, srcs[t], [r["id"] for r in data[t]])
    assert len(mix) == len(emb["minilm"]["mix"]), (len(mix), len(emb["minilm"]["mix"]))

    tuned = json.loads((RUN / "out" / args.tag / "tuned_params_repro_two_sided.json").read_text())["params"]
    params = {a: v for a, v in tuned.items() if a != "tfidf1_lr_sohan"}
    fixed = compute_arms(params, mix, ref, tests, emb, lengths)
    sohan = compute_arms({"tfidf1_lr_sohan": {}}, [r["text"] for r in idfit_rows], ref, tests, {}, lengths)
    for p in fixed:
        fixed[p]["tfidf1_lr_sohan"] = sohan[p]["tfidf1_lr_sohan"]
    arms = [a for a in ALL_ARMS if a in fixed["deploy"]]
    print(f"fixed scores for {len(arms)} arms in {time.time() - t0:.0f}s", flush=True)

    out = RUN / "out" / args.tag
    meta = []
    for p, rows in fixed_pools.items():
        for i, r in enumerate(rows):
            meta.append({"id": r["id"], "type": p if p in TYPES else "deploy",
                         "source": "needle" if p in TYPES else ("lmsys" if r["id"].startswith("lmsys") else "wildchat"),
                         "n_tokens": int(lengths[p][i]), **{a: float(fixed[p][a][i]) for a in arms}})
    write_jsonl(meta, RUN / "scores" / args.tag / "ranking_ranking_variants_fixed.jsonl")

    res, cluster_res, tops = [], [], {}
    for s in NEEDLE_SEEDS:
        parts = [("deploy", np.arange(len(bg)))] + [(t, np.arange(s * n_per, (s + 1) * n_per)) for t in TYPES]
        text = [tests[n][i] for n, idx in parts for i in idx]
        rid = [fixed_pools[n][i]["id"] for n, idx in parts for i in idx]
        types = np.array([("deploy" if n == "deploy" else n) for n, idx in parts for _ in idx])
        L = np.concatenate([lengths[n][idx] for n, idx in parts])
        sc = {a: np.concatenate([fixed[n][a][idx] for n, idx in parts]) for a in arms}
        sc["random"] = np.random.default_rng(1000 + s).random(len(L))  # run_ranking_bench.py:194
        sc["tfidf1_dr"] = cross_fit_tfidf(text, mix, 1.0, 5, s)
        write_jsonl([{"id": rid[i], "type": str(types[i]), "tfidf1_dr": float(sc["tfidf1_dr"][i])} for i in range(len(L))],
                    RUN / "scores" / args.tag / f"ranking_ranking_variants_dr_seed{s}.jsonl")
        all_arms = [*arms, "tfidf1_dr", "random"]
        pct = {a: length_controlled(sc[a], L) for a in all_arms}
        for a in all_arms:
            for ranking, v in (("raw", sc[a]), ("pct20", pct[a])):
                order = jitter_order(v, s)
                m = top_metrics(order, types, L, TOP_K)
                top = order[:TOP_K]
                m["top_name_placeholder_share"] = float(np.mean([bool(re.search(r"NAME_\d", text[i])) for i in top]))
                m["top_lmsys_share"] = float(np.mean([rid[i].startswith("lmsys") for i in top]))
                m["top_non_ascii_share"] = float(np.mean([sum(ord(c) > 127 for c in text[i]) > 0.1 * len(text[i]) for i in top]))
                res.append({"needle_seed": s, "arm": a, "side": "one" if a in ONE_SIDED else ("random" if a == "random" else "two"),
                            "ranking": ranking, **m})
                if s == 0:
                    tops[f"{a}|{ranking}"] = [rid[i] for i in top]
        # whole-cluster review (ranking_variants Q3): clusters on multilingual MiniLM and on tf-idf SVD-256, ordered by mean pct20 score
        reps = {"mminilm": np.concatenate([emb["mminilm"][n][idx] for n, idx in parts])}
        v = TfidfVectorizer(sublinear_tf=True, min_df=2, max_features=200_000).fit(text)
        reps["tfidf_svd256"] = TruncatedSVD(256 if not lim else 32, random_state=0).fit_transform(v.transform(text))
        k = KMEANS_K if not lim else 20
        for rn, x in reps.items():
            labels = kmeans_labels(x, k, 0)
            write_jsonl([{"id": rid[i], "cluster": int(labels[i])} for i in range(len(L))],
                        RUN / "scores" / args.tag / f"ranking_ranking_variants_clusters_{rn}_seed{s}.jsonl")
            for a in all_arms:
                order = cluster_order(pct[a], labels, TOP_K)
                cluster_res.append({"needle_seed": s, "arm": a, "cluster_rep": rn, "kmeans_k": k,
                                    "ranking": "pct20_cluster", **top_metrics(order, types, L, TOP_K)})
        print(f"seed {s} done {time.time() - t0:.0f}s", flush=True)

    for name, rows in (("ranking_ranking_variants_per_seed", res), ("ranking_ranking_variants_cluster_per_seed", cluster_res)):
        with open(out / f"{name}.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    # mean over needle seeds
    summ = []
    keys = ["gap_uplift@100", "legal_uplift@100", "medical_uplift@100", "control_uplift@100", "top_median_len_ratio"]
    for rows, by in ((res, ("arm", "ranking")), (cluster_res, ("arm", "cluster_rep", "ranking"))):
        groups = {}
        for r in rows:
            groups.setdefault(tuple(r[b] for b in by), []).append(r)
        for g, rr in groups.items():
            summ.append({**dict(zip(by, g)), **{f"mean_{k}": float(np.mean([r[k] for r in rr])) for k in keys},
                         "per_seed_gap_uplift": [r["gap_uplift@100"] for r in rr],
                         "per_seed_len_ratio": [r["top_median_len_ratio"] for r in rr]})
    write_json(summ, out / "ranking_ranking_variants_summary.json")
    write_json(tops, out / "ranking_ranking_variants_top100_ids_seed0.json")
    for r in sorted([x for x in summ if x["ranking"] == "pct20"], key=lambda x: -x["mean_gap_uplift@100"])[:12]:
        print(f"{r['arm']:16s} pct20 gap {r['mean_gap_uplift@100']:.2f} legal {r['mean_legal_uplift@100']:.2f} "
              f"medical {r['mean_medical_uplift@100']:.2f} control {r['mean_control_uplift@100']:.2f} len {r['per_seed_len_ratio']}")
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
