# known-gap tasks (Sohan's repro_two_sided / repro_olmo2_1b): legal and medical prompts vs English WildChat, raw and length-stratified
# AUROC with 1,000-resample bootstrap intervals, for every word baseline at its validation-tuned setting.
# Comparisons and statistics as in scripts/run_detectors.py:26-33 and src/metrics/detectors.py:63-104.
# usage: .venv/bin/python -m scripts.wb_known_gap --exp repro_two_sided [--limit 200 --tag tiny]

import argparse
import csv
import json
import re
import time

import numpy as np

from scripts.wb_common import INPUTS, RUN, auroc, boot_indices, load_jsonl, stratified_auroc, write_json, write_jsonl
from scripts.wb_detectors import ALL_ARMS, ONE_SIDED, compute_arms, make_vectorizer, topk_sims

TEST_POOLS = ("idmix", "legal", "medical", "wildchat_en_test")
COMPARISONS = [("legal", "idmix"), ("medical", "idmix"), ("wildchat_en_test", "idmix"),
               ("legal", "wildchat_en_test"), ("medical", "wildchat_en_test")]
POOL_FILE = {"repro_two_sided": {p: f"repro_two_sided/{p}" for p in (*TEST_POOLS, "idfit", "dolma")},
             "repro_olmo2_1b": {**{p: f"repro_olmo2_1b/{p}" for p in ("idfit", "idmix", "wildchat_en_test")},
                       "legal": "repro_two_sided/legal", "medical": "repro_two_sided/medical", "dolma": "repro_two_sided/dolma"}}
PATTERNS = {
    "name_placeholder": r"NAME_\d", "markdown_fence": r"```", "markdown_bold": r"\*\*", "url": r"https?://",
    "response_constraint": r"(?i)your (entire )?response (should|must)", "jurisdiction_tag": r"^\s*[\[(][A-Z]{2,}[\])]",
    "hi_doctor": r"(?i)^\s*(hi|hello|dear)\b.{0,20}doc", "stripped_apostrophe": r"\b(I m|don t|can t|it s)\b",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--tag", default="full")
    ap.add_argument("--n_boot", type=int, default=1000)
    args = ap.parse_args()
    t0 = time.time()
    lim = args.limit
    files = POOL_FILE[args.exp]
    rows = {p: load_jsonl(INPUTS / f"{files[p]}.jsonl")[:lim] for p in files}
    texts = {p: [r["text"] for r in rows[p]] for p in rows}
    ref = [" ".join(t.split()[:400]) for t in texts["dolma"]]  # scripts/run_detectors.py:116
    ids = {p: [r["id"] for r in rows[p]] for p in rows}
    lengths = {}
    for p in TEST_POOLS:
        L = json.loads((RUN / "cache" / "lengths" / args.exp / f"{p}.json").read_text())["lengths"]
        lengths[p] = np.array([L[i] for i in ids[p]])
    emb = {}
    for e in ("minilm", "mminilm"):
        emb[e] = {}
        for key, p in (("mix", "idfit"), ("ref", "dolma"), *((q, q) for q in TEST_POOLS)):
            f = np.load(RUN / "cache" / f"emb_{args.tag}" / e / f"{files[p].replace('/', '__')}.npz")
            assert list(f["ids"][: len(ids[p])]) == ids[p], (e, p)
            emb[e][key] = f["emb"][: len(ids[p])]
    tuned = json.loads((RUN / "out" / args.tag / f"tuned_params_{args.exp}.json").read_text())
    params = tuned["params"]
    tests = {p: texts[p] for p in TEST_POOLS}
    scores, models = compute_arms(params, texts["idfit"], ref, tests, emb, lengths, return_models=True)
    arms = [a for a in ALL_ARMS if a in scores["idmix"]]
    print(f"scored {len(arms)} arms in {time.time() - t0:.0f}s", flush=True)

    out = RUN / "out" / args.tag
    # per-prompt raw scores
    write_jsonl([{"id": ids[p][i], "pool": p, "n_content_tokens": int(lengths[p][i]),
                  **{a: float(scores[p][a][i]) for a in arms}, "text": texts[p][i]}
                 for p in TEST_POOLS for i in range(len(ids[p]))], RUN / "scores" / args.tag / f"known_gap_{args.exp}.jsonl")

    # AUROC tables; one set of bootstrap draws per comparison, shared by all arms (paired)
    table, bands = [], {}
    deltas = []
    for o, r in COMPARISONS:
        draws = boot_indices(len(ids[o]), len(ids[r]), args.n_boot, 0)
        lo, lr = lengths[o], lengths[r]
        boot_strat = {}
        for a in arms:
            so, sr = scores[o][a], scores[r][a]
            st = stratified_auroc(so, sr, lo, lr)
            braw = [auroc(so[i], sr[j]) for i, j in draws]
            bst = [stratified_auroc(so[i], sr[j], lo[i], lr[j])["auroc"] for i, j in draws]
            boot_strat[a] = np.array([np.nan if v is None else v for v in bst])
            ok = [v for v in bst if v is not None]
            table.append({
                "exp": args.exp, "arm": a, "side": "one" if a in ONE_SIDED else "two", "ood": o, "ref": r,
                "auroc_raw": auroc(so, sr), "raw_lo": float(np.percentile(braw, 2.5)), "raw_hi": float(np.percentile(braw, 97.5)),
                "auroc_strat": st["auroc"],
                "strat_lo": float(np.percentile(ok, 2.5)) if len(ok) >= 0.9 * args.n_boot else None,
                "strat_hi": float(np.percentile(ok, 97.5)) if len(ok) >= 0.9 * args.n_boot else None,
                "strat_n_bands": st["n_bands"], "strat_coverage": st["coverage"], "n_ood": len(so), "n_ref": len(sr),
                "val_objective": tuned["best_objective"].get(a),
            })
            bands[f"{a}|{o}|{r}"] = st["bands"]
        if r == "wildchat_en_test":
            base = boot_strat["tfidf1_lr_sohan"]
            for a in arms:
                d = boot_strat[a] - base
                d = d[~np.isnan(d)]
                pt = [t for t in table if (t["arm"], t["ood"], t["ref"]) == (a, o, r)][0]["auroc_strat"]
                pb = [t for t in table if (t["arm"], t["ood"], t["ref"]) == ("tfidf1_lr_sohan", o, r)][0]["auroc_strat"]
                deltas.append({"exp": args.exp, "arm": a, "ood": o, "delta_vs_sohan_tfidf": pt - pb,
                               "lo": float(np.percentile(d, 2.5)), "hi": float(np.percentile(d, 97.5))})
        print(f"{o} vs {r} done {time.time() - t0:.0f}s", flush=True)
    with open(out / f"known_gap_{args.exp}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(table[0]))
        w.writeheader()
        w.writerows(table)
    write_json(bands, out / f"known_gap_{args.exp}_bands.json")
    write_json(deltas, out / f"known_gap_{args.exp}_paired_delta_vs_sohan_tfidf.json")

    # checks: near-duplicates to the fit set, format patterns, top logistic features
    v = make_vectorizer("tfidf1", min_df=1, sublinear=True).fit(texts["idfit"])
    xf = v.transform(texts["idfit"])
    dup = {}
    for p in TEST_POOLS:
        m = topk_sims(v.transform(tests[p]), xf, 1)[:, 0]
        dup[p] = {"max_cos_to_idfit_median": float(np.median(m)), "n_ge_0.8": int((m >= 0.8).sum()),
                  "n_ge_0.9": int((m >= 0.9).sum()), "n": len(m),
                  "examples_ge_0.9": [ids[p][i] for i in np.where(m >= 0.9)[0][:10]]}
    pats = {p: {k: float(np.mean([bool(re.search(rx, t, flags=re.M)) for t in texts[p]])) for k, rx in PATTERNS.items()}
            for p in (*TEST_POOLS, "idfit")}
    for p in pats:
        pats[p]["non_ascii_share_rows_gt_10pct"] = float(np.mean([sum(ord(c) > 127 for c in t) > 0.1 * max(len(t), 1) for t in texts[p]]))
    feats = {}
    for a, (vec, clf) in models.items():
        names = np.array(vec.get_feature_names_out())
        coef = clf.coef_[0]
        o = np.argsort(coef)
        feats[a] = {"toward_mix (low score)": names[o[:30]].tolist(), "toward_dolma (high score)": names[o[-30:][::-1]].tolist()}
    write_json({"near_duplicates": dup, "pattern_rates": pats, "top_lr_features": feats}, out / f"checks_{args.exp}.json")

    # reading: 20 random per set, 20 top / bottom of the validation-best arm over legal + medical + wildchat
    rng = np.random.default_rng(0)
    read = {"random": {p: [texts[p][i][:600] for i in rng.choice(len(texts[p]), min(20, len(texts[p])), replace=False)]
                       for p in (*TEST_POOLS, "idfit")}}
    cand = {a: tuned["best_objective"][a] for a in arms if a in tuned["best_objective"]}
    best = max(cand, key=cand.get)
    pool_rows = [(p, i) for p in ("legal", "medical", "wildchat_en_test") for i in range(len(ids[p]))]
    s = np.array([scores[p][best][i] for p, i in pool_rows])
    o = np.argsort(-s)
    fmt = lambda k: {"pool": pool_rows[k][0], "id": ids[pool_rows[k][0]][pool_rows[k][1]], "score": float(s[k]),
                     "n_tokens": int(lengths[pool_rows[k][0]][pool_rows[k][1]]), "text": texts[pool_rows[k][0]][pool_rows[k][1]][:600]}
    read["best_arm_by_validation"] = best
    read["top20"] = [fmt(k) for k in o[:20]]
    read["bottom20"] = [fmt(k) for k in o[-20:]]
    # within-pool extremes that matter for label noise: lowest-scored gap rows, highest-scored wildchat rows
    for p in ("legal", "medical", "wildchat_en_test"):
        idx = [k for k in range(len(pool_rows)) if pool_rows[k][0] == p]
        oo = sorted(idx, key=lambda k: -s[k])
        read[f"{p}_top10"] = [fmt(k) for k in oo[:10]]
        read[f"{p}_bottom10"] = [fmt(k) for k in oo[-10:]]
    write_json(read, out / "read" / f"read_{args.exp}.json")
    print(f"done {args.exp} in {time.time() - t0:.0f}s; best arm by validation {best}")


if __name__ == "__main__":
    main()
