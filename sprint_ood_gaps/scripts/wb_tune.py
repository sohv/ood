# tunes every word baseline on the validation pools only (fresh legal / medical vs shard-7 English WildChat with the
# mix's members removed). Objective: mean of the two length-stratified AUROCs (legal vs val WildChat, medical vs val
# WildChat), the same statistic as the test comparison. Test pools are never loaded here.
# usage: .venv/bin/python -m scripts.wb_tune --exp exp01 [--limit 200 --tag tiny]

import argparse
import itertools
import time

import numpy as np

from scripts.wb_common import INPUTS, RUN, load_jsonl, stratified_auroc, write_json
from scripts.wb_detectors import REPS, UnigramLM, _counts, knn_dist, lr_scores, make_vectorizer, maha_scores, topk_sims

MIX_OF = {"exp01": "dolci", "exp12": "tulu"}
KS = (1, 3, 5, 10, 25, 50, 100, 200)
CS_SPARSE = (0.001, 0.003, 0.01, 0.1, 1.0, 10.0)
CS_DENSE = (0.01, 0.1, 1.0, 10.0, 100.0)
SHRINK = ("lw", 0.01, 0.1, 0.5, 0.9)
ALPHAS = (0.01, 0.1, 0.5, 1.0, 3.0)


def vec_grid(rep):
    if rep == "char35":
        return [dict(min_df=m, sublinear=s, analyzer=a) for a in ("char_wb", "char") for m in (2, 5) for s in (True, False)]
    return [dict(min_df=m, sublinear=s) for m in (1, 2, 5) for s in (True, False)]


def load_lengths(exp, pool):
    import json

    return json.loads((RUN / "cache" / "lengths" / exp / f"{pool}.json").read_text())["lengths"]


def load_emb(tag, name, pool, ids):
    f = np.load(RUN / "cache" / f"emb_{tag}" / name / f"{pool.replace('/', '__')}.npz")
    assert list(f["ids"][: len(ids)]) == ids, f"{name} {pool} ids misaligned"
    return f["emb"][: len(ids)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--tag", default="full")
    args = ap.parse_args()
    lim = args.limit
    mixname = MIX_OF[args.exp]
    t0 = time.time()
    rows = {
        "mix": load_jsonl(INPUTS / args.exp / "idfit.jsonl")[:lim],
        "ref": load_jsonl(INPUTS / "exp01" / "dolma.jsonl")[:lim],
        "legal": load_jsonl(INPUTS / "val" / "val_legal.jsonl")[:lim],
        "medical": load_jsonl(INPUTS / "val" / "val_medical.jsonl")[:lim],
        "wc": load_jsonl(INPUTS / "val" / f"val_wildchat_{mixname}.jsonl")[:lim],
    }
    texts = {k: [r["text"] for r in v] for k, v in rows.items()}
    texts["ref"] = [" ".join(t.split()[:400]) for t in texts["ref"]]
    ids = {k: [r["id"] for r in v] for k, v in rows.items()}
    lens = {}
    for k, pool in (("legal", "val_legal"), ("medical", "val_medical"), ("wc", f"val_wildchat_{mixname}")):
        L = load_lengths(f"val_{args.exp}", pool)
        lens[k] = np.array([L[i] for i in ids[k]])
    tests = ("legal", "medical", "wc")

    def objective(sc):
        vals = [stratified_auroc(sc[p], sc["wc"], lens[p], lens["wc"])["auroc"] for p in ("legal", "medical")]
        return None if any(v is None for v in vals) else float(np.mean(vals)), vals

    grid = []

    def record(arm, hp, sc):
        obj, vals = objective(sc)
        grid.append({"arm": arm, **{f"hp_{k}": v for k, v in hp.items()}, "objective": obj,
                     "val_legal": vals[0], "val_medical": vals[1]})

    grid.append({"arm": "length", "objective": objective({p: lens[p].astype(float) for p in tests})[0]})

    # sparse representations
    for rep in REPS:
        for vh in vec_grid(rep):
            v = make_vectorizer(rep, **vh)
            xm = v.fit_transform(texts["mix"])
            tops = {p: topk_sims(v.transform(texts[p]), xm) for p in tests}
            for k in KS:
                record(f"{rep}_knn", {**vh, "k": k}, {p: knn_dist(tops[p], k) for p in tests})
            v2 = make_vectorizer(rep, **vh).fit(texts["mix"] + texts["ref"])
            xm2, xr2 = v2.transform(texts["mix"]), v2.transform(texts["ref"])
            xt = {p: v2.transform(texts[p]) for p in tests}
            tm = {p: topk_sims(xt[p], xm2) for p in tests}
            tr = {p: topk_sims(xt[p], xr2) for p in tests}
            for k in KS:
                record(f"{rep}_dknn", {**vh, "k": k}, {p: knn_dist(tm[p], k) - knn_dist(tr[p], k) for p in tests})
            for c in CS_SPARSE:
                sc, _ = lr_scores(xm2, xr2, xt, c)
                record(f"{rep}_lr", {**vh, "C": c}, sc)
            print(f"{rep} {vh} done {time.time() - t0:.0f}s", flush=True)

    # dense embeddings
    pool_of = {"mix": f"{args.exp}/idfit", "ref": "exp01/dolma", "legal": "val/val_legal", "medical": "val/val_medical",
               "wc": f"val/val_wildchat_{mixname}"}
    for e in ("minilm", "mminilm"):
        E = {k: load_emb(args.tag, e, pool_of[k], ids[k]) for k in pool_of}
        tm = {p: topk_sims(E[p], E["mix"]) for p in tests}
        tr = {p: topk_sims(E[p], E["ref"]) for p in tests}
        for k in KS:
            record(f"{e}_knn", {"k": k}, {p: knn_dist(tm[p], k) for p in tests})
            record(f"{e}_dknn", {"k": k}, {p: knn_dist(tm[p], k) - knn_dist(tr[p], k) for p in tests})
        for c in CS_DENSE:
            sc, _ = lr_scores(E["mix"], E["ref"], {p: E[p] for p in tests}, c)
            record(f"{e}_lr", {"C": c}, sc)
        for s in SHRINK:
            m = {p: maha_scores(E["mix"], E[p], s) for p in tests}
            r = {p: maha_scores(E["ref"], E[p], s) for p in tests}
            record(f"{e}_maha", {"shrinkage": s}, m)
            record(f"{e}_dmaha", {"shrinkage": s}, {p: m[p] - r[p] for p in tests})
        print(f"{e} done {time.time() - t0:.0f}s", flush=True)

    vocab = len(set(_counts(texts["mix"])) | set(_counts(texts["ref"])))
    for a in ALPHAS:
        lm = UnigramLM(texts["mix"], a)
        record("uni_lm", {"alpha": a}, {p: lm.nll(texts[p]) for p in tests})
        lm_m, lm_r = UnigramLM(texts["mix"], a, vocab), UnigramLM(texts["ref"], a, vocab)
        record("uni_nb", {"alpha": a}, {p: lm_m.nll(texts[p]) - lm_r.nll(texts[p]) for p in tests})

    # pick the best setting per arm (first in grid order on ties)
    best = {}
    for g in grid:
        if g["arm"] == "length" or g["objective"] is None:
            continue
        if g["arm"] not in best or g["objective"] > best[g["arm"]]["objective"]:
            best[g["arm"]] = g
    params = {a: {k[3:]: v for k, v in g.items() if k.startswith("hp_")} for a, g in best.items()}
    params["tfidf1_lr_sohan"] = {"fixed": "min_df 2, sublinear, unigram, C 0.1 (src/metrics/detectors.py:55-60)"}
    out = RUN / "out" / args.tag
    write_json(grid, out / f"tuning_grid_{args.exp}.json")
    write_json({"params": params, "best_objective": {a: best[a]["objective"] for a in best},
                "n_settings": {a: sum(g["arm"] == a for g in grid) for a in best}}, out / f"tuned_params_{args.exp}.json")
    for a in sorted(best, key=lambda a: -best[a]["objective"]):
        print(f"{a:16s} {best[a]['objective']:.3f}  legal {best[a]['val_legal']:.3f} medical {best[a]['val_medical']:.3f}  {params[a]}")
    print(f"tuning {args.exp} done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
