# exp10: scores word-banned rewrites with the exp01 detectors, tests probe transfer from originals, and the template-wrapped dolma ablation.
# uv run -m scripts.run_lexical_detectors --run_dir results/raw/261019_exp10_lexical_v1 --exp01_pools data/processed/exp01 --exp10_pools data/processed/exp10 --seed 0

import os

os.environ.setdefault("OMP_NUM_THREADS", "16")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "16")

import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import simple_parsing
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.data.io import load_jsonl
from src.interp.features import LAYERS, POOLINGS
from src.metrics.detectors import KNN, LENGTH_BANDS, Mahalanobis, auroc, stratified_auroc, tfidf_mix_vs_dolma
from src.metrics.io import write_json, write_jsonl
from src.utils.git import get_git_hash
from src.utils.logging import setup_logging
from src.utils.seed import set_seed

LOGGER = logging.getLogger(__name__)

# identical to exp04's bands: log-spaced by 2^(1/4) from 1 to 1024 content tokens
FINE_BANDS = tuple(float(b) for b in np.unique(np.round(2 ** (np.arange(0, 41) / 4)))) + (1025.0,)
ORIG_POOLS = ["legal", "medical", "wildchat_en_test", "idmix"]
LX_POOLS = ["legal_lx", "medical_lx", "wildchat_lx"]
DOLMA = ("full", "matched", "wrapped")
BAR_DETECTORS = ("dmaha_full", "dknn_full")
DETECTORS = ("maha", "knn", "dmaha_full", "dknn_full", "dmaha_matched", "dknn_matched", "dmaha_wrapped", "dknn_wrapped", "tfidf", "length")
PRIMARY = [("legal_lx", "wildchat_lx"), ("medical_lx", "wildchat_lx")]
ORIGINAL = [("legal", "wildchat_en_test"), ("medical", "wildchat_en_test")]
EXP01_DMAHA_FULL_COARSE = {"legal": 0.954, "medical": 0.897}
TARGET_TOPIC = {"legal_lx": "lawyer", "medical_lx": "doctor"}


@dataclass
class LexicalDetectorConfig:
    run_dir: str = ""
    exp01_pools: str = ""
    exp10_pools: str = ""
    seed: int = 0
    n_boot: int = 1000
    pooling: str = "cmean"
    layer: int = 16
    dolma_tfidf_words: int = 400
    n_folds: int = 5


def fit_detectors(x: dict[str, np.ndarray]):
    scale = x["idfit"].std(0) + 1e-6
    maha_mix, knn_mix = Mahalanobis(x["idfit"]), KNN(x["idfit"], scale)
    dolma = {v: (Mahalanobis(x[f"dolma_{v}"]), KNN(x[f"dolma_{v}"], scale)) for v in DOLMA}

    def score(feats: np.ndarray) -> dict[str, np.ndarray]:
        m, k = maha_mix(feats), knn_mix(feats)
        out = {"maha": m, "knn": k}
        for v, (md, kd) in dolma.items():
            out[f"dmaha_{v}"], out[f"dknn_{v}"] = m - md(feats), k - kd(feats)
        return out

    return score


def paired_boot(stat_pair, n_o: int, n_r: int, n_boot: int, rng) -> list[tuple]:
    return [stat_pair(rng.integers(0, n_o, n_o), rng.integers(0, n_r, n_r)) for _ in range(n_boot)]


def ci(values: list[float | None], level: float) -> list[float] | None:
    values = [v for v in values if v is not None]
    return [float(np.percentile(values, (100 - level) / 2)), float(np.percentile(values, 100 - (100 - level) / 2))] if values else None


def main():
    config = simple_parsing.parse(LexicalDetectorConfig)
    run = Path(config.run_dir)
    setup_logging(run)
    set_seed(config.seed)
    write_json({"git_hash": get_git_hash(), **asdict(config), "fine_bands": FINE_BANDS, "coarse_bands": LENGTH_BANDS}, run / "detector_config.json")

    feat_names = ["idfit", *(f"dolma_{v}" for v in DOLMA), *ORIG_POOLS, *LX_POOLS]
    feats = {n: dict(np.load(run / "features" / f"{n}.npz")) for n in feat_names}
    rows = {n: load_jsonl(Path(config.exp01_pools) / f"{n}.jsonl") for n in ["idfit", "dolma", *ORIG_POOLS]}
    rows |= {n: load_jsonl(Path(config.exp10_pools) / f"{n}.jsonl") for n in LX_POOLS}
    for n in ["idfit", *ORIG_POOLS, *LX_POOLS]:
        assert [str(i) for i in feats[n]["ids"]] == [r["id"] for r in rows[n]], f"{n} rows misaligned"
    # every rewrite pool row must point at the original at the same index, for the cross-fitted transfer test
    for lx, orig in zip(LX_POOLS, ["legal", "medical", "wildchat_en_test"]):
        assert [r["source_id"] for r in rows[lx]] == [r["id"] for r in rows[orig]], f"{lx} does not align with {orig}"
    texts = {n: [r["text"] for r in rs] for n, rs in rows.items()}
    lengths = {n: f["n_content_tokens"] for n, f in feats.items()}
    on_topic = {n: np.array([r["topic_rewrite"] == TARGET_TOPIC.get(n, "neither") for r in rows[n]]) for n in LX_POOLS}
    leak_free = {n: np.array([not r["leaks"] for r in rows[n]]) for n in LX_POOLS}

    tfidf = tfidf_mix_vs_dolma(texts["idfit"], [" ".join(t.split()[: config.dolma_tfidf_words]) for t in texts["dolma"]], config.seed)
    test_pools = [*ORIG_POOLS, *LX_POOLS]
    tfidf_scores = {n: tfidf(texts[n]) for n in test_pools}

    def score_cell(pooling: str, layer: int) -> dict[str, dict[str, np.ndarray]]:
        li = LAYERS.index(layer)
        x = {n: f[pooling][:, li].astype(np.float32) for n, f in feats.items()}
        score = fit_detectors(x)
        return {n: score(x[n]) | {"tfidf": tfidf_scores[n], "length": lengths[n].astype(float)} for n in test_pools}

    scores = score_cell(config.pooling, config.layer)
    LOGGER.info(f"scored decision cell {config.pooling} L{config.layer}")

    def strat(det: str, ood: str, ref: str, i=None, j=None, sc=None, bands=FINE_BANDS) -> float | None:
        sc = sc or scores
        o, r, lo, lr = sc[ood][det], sc[ref][det], lengths[ood], lengths[ref]
        if i is not None:
            o, lo, r, lr = o[i], lo[i], r[j], lr[j]
        return stratified_auroc(o, r, lo, lr, bands)["auroc"]

    # reproduction check against exp01 before anything else is read
    repro = {p: strat("dmaha_full", p, "wildchat_en_test", bands=LENGTH_BANDS) for p in EXP01_DMAHA_FULL_COARSE}
    print(f"{repro=}")
    for p, v in repro.items():
        assert abs(v - EXP01_DMAHA_FULL_COARSE[p]) <= 0.01, f"re-extracted features do not reproduce exp01 on {p}: {v} vs {EXP01_DMAHA_FULL_COARSE[p]}"

    rng = np.random.default_rng(config.seed)
    primary = {}
    for ood, ref in PRIMARY:
        n_o, n_r = len(scores[ood]["length"]), len(scores[ref]["length"])
        boots = [(rng.integers(0, n_o, n_o), rng.integers(0, n_r, n_r)) for _ in range(config.n_boot)]
        point = {d: strat(d, ood, ref) for d in DETECTORS}
        boot_vals = {d: [strat(d, ood, ref, i, j) for i, j in boots] for d in (*BAR_DETECTORS, "tfidf")}
        entry = {"auroc_stratified": point, "auroc_raw": {d: auroc(scores[ood][d], scores[ref][d]) for d in DETECTORS},
                 "length_only_stratified": point["length"], "balanced": point["length"] is not None and 0.45 <= point["length"] <= 0.55,
                 "tfidf_ci": ci(boot_vals["tfidf"], 98.75)}
        for d in BAR_DETECTORS:
            deltas = [a - t for a, t in zip(boot_vals[d], boot_vals["tfidf"]) if a is not None and t is not None]
            entry[f"{d}_ci"] = ci(boot_vals[d], 98.75)
            entry[f"delta_{d}_minus_tfidf"] = None if point[d] is None or point["tfidf"] is None else point[d] - point["tfidf"]
            entry[f"delta_{d}_ci"] = ci(deltas, 98.75) if len(deltas) >= 0.9 * config.n_boot else None
        primary[f"{ood}_vs_{ref}"] = entry
        LOGGER.info(f"{ood} vs {ref}: {entry}")

    topic_rates = {n: float(on_topic[n].mean()) for n in TARGET_TOPIC}
    g1 = all(v >= 0.80 for v in topic_rates.values())
    g2_detail = {k: e["auroc_stratified"]["tfidf"] for k, e in primary.items()}
    g2 = all(v is not None and v <= 0.65 for v in g2_detail.values())
    g3 = all(e["balanced"] for e in primary.values())

    def adds(d: str) -> bool:
        return all(e["auroc_stratified"][d] is not None and e["auroc_stratified"][d] >= 0.75 and e[f"delta_{d}_minus_tfidf"] >= 0.10
                   and e[f"delta_{d}_ci"] is not None and e[f"delta_{d}_ci"][0] > 0 for e in primary.values())

    def words_only(d: str) -> bool:
        return all(e["auroc_stratified"][d] is not None and (e["auroc_stratified"][d] <= 0.65 or (e[f"delta_{d}_ci"] is not None and e[f"delta_{d}_ci"][1] < 0.10))
                   for e in primary.values())

    if not (g1 and g3):
        q1_verdict = f"inconclusive: gate failed (G1 topic kept {g1}, G3 balance {g3})"
    elif any(adds(d) for d in BAR_DETECTORS):
        q1_verdict = "activations_add_beyond_words"
    elif all(words_only(d) for d in BAR_DETECTORS):
        q1_verdict = "words_only" if g2 else "inconclusive: words-only pattern but G2 failed (tf-idf still separates the rewrites)"
    else:
        q1_verdict = "ambiguous" + ("" if g2 else ", and G2 failed (tf-idf still separates the rewrites)")

    # q2: probes trained on originals, scored on rewrites of held-out rows
    li = LAYERS.index(config.layer)
    act = {n: feats[n][config.pooling][:, li].astype(np.float32) for n in test_pools}
    encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cuda")
    emb = {n: encoder.encode(texts[n], batch_size=128, normalize_embeddings=True) for n in ["legal", "medical", "wildchat_en_test", *LX_POOLS]}

    def act_probe(xa, xb):
        scaler = StandardScaler().fit(np.r_[xa, xb])
        clf = LogisticRegression(C=0.1, max_iter=5000, random_state=config.seed).fit(scaler.transform(np.r_[xa, xb]), np.r_[np.ones(len(xa)), np.zeros(len(xb))])
        return lambda x: clf.decision_function(scaler.transform(x))

    def tfidf_probe(ta, tb):
        vec = TfidfVectorizer(sublinear_tf=True, min_df=2, max_features=200_000)
        clf = LogisticRegression(C=1.0, max_iter=2000, random_state=config.seed).fit(vec.fit_transform(ta + tb), np.r_[np.ones(len(ta)), np.zeros(len(tb))])
        return lambda t: clf.decision_function(vec.transform(t))

    def emb_probe(xa, xb):
        clf = LogisticRegression(C=1.0, max_iter=5000, random_state=config.seed).fit(np.r_[xa, xb], np.r_[np.ones(len(xa)), np.zeros(len(xb))])
        return lambda x: clf.decision_function(x)

    def fold_of(n: int) -> np.ndarray:
        return np.random.default_rng(config.seed).permutation(n) % config.n_folds

    transfer, probe_scores = {}, {}
    for p, p_lx in (("legal", "legal_lx"), ("medical", "medical_lx")):
        fa, fb = fold_of(len(texts[p])), fold_of(len(texts["wildchat_en_test"]))
        out = {arm: {"lx_p": np.zeros(len(fa)), "lx_w": np.zeros(len(fb)), "orig_p": np.zeros(len(fa)), "orig_w": np.zeros(len(fb))} for arm in ("act", "tfidf", "minilm")}
        for k in range(config.n_folds):
            tr_a, tr_b, te_a, te_b = np.where(fa != k)[0], np.where(fb != k)[0], np.where(fa == k)[0], np.where(fb == k)[0]
            probes = {
                "act": (act_probe(act[p][tr_a], act["wildchat_en_test"][tr_b]), act),
                "tfidf": (tfidf_probe([texts[p][i] for i in tr_a], [texts["wildchat_en_test"][i] for i in tr_b]), None),
                "minilm": (emb_probe(emb[p][tr_a], emb["wildchat_en_test"][tr_b]), emb),
            }
            for arm, (f, src) in probes.items():
                for key, pool, idx in (("lx_p", p_lx, te_a), ("lx_w", "wildchat_lx", te_b), ("orig_p", p, te_a), ("orig_w", "wildchat_en_test", te_b)):
                    out[arm][key][idx] = f([texts[pool][i] for i in idx]) if src is None else f(src[pool][idx])
        probe_scores[p] = out

        def st(arm: str, kind: str, i=None, j=None) -> float | None:
            a_pool, b_pool = (p_lx, "wildchat_lx") if kind == "lx" else (p, "wildchat_en_test")
            o, r, lo, lr = out[arm][f"{kind}_p"], out[arm][f"{kind}_w"], lengths[a_pool], lengths[b_pool]
            if i is not None:
                o, lo, r, lr = o[i], lo[i], r[j], lr[j]
            return stratified_auroc(o, r, lo, lr, FINE_BANDS)["auroc"]

        n_o, n_r = len(fa), len(fb)
        boots = [(rng.integers(0, n_o, n_o), rng.integers(0, n_r, n_r)) for _ in range(config.n_boot)]
        bv = {arm: [st(arm, "lx", i, j) for i, j in boots] for arm in ("act", "tfidf", "minilm")}
        diffs = [a - t for a, t in zip(bv["act"], bv["tfidf"]) if a is not None and t is not None]
        point = {arm: st(arm, "lx") for arm in ("act", "tfidf", "minilm")}
        transfer[p] = {
            "transfer_stratified": point,
            "transfer_raw": {arm: auroc(out[arm]["lx_p"], out[arm]["lx_w"]) for arm in out},
            "transfer_ci": {arm: ci(bv[arm], 97.5) for arm in bv},
            "act_minus_tfidf": None if point["act"] is None or point["tfidf"] is None else point["act"] - point["tfidf"],
            "act_minus_tfidf_ci": ci(diffs, 97.5) if len(diffs) >= 0.9 * config.n_boot else None,
            "act_minus_minilm": None if point["act"] is None or point["minilm"] is None else point["act"] - point["minilm"],
            "in_distribution_stratified": {arm: st(arm, "orig") for arm in out},
            "in_distribution_raw": {arm: auroc(out[arm]["orig_p"], out[arm]["orig_w"]) for arm in out},
        }
        LOGGER.info(f"transfer {p}: {transfer[p]}")
    t_ok = [e["act_minus_tfidf"] is not None and e["act_minus_tfidf_ci"] is not None for e in transfer.values()]
    if all(t_ok) and all(e["act_minus_tfidf"] >= 0.10 and e["act_minus_tfidf_ci"][0] > 0 for e in transfer.values()):
        q2_verdict = "activations_transfer_beyond_words"
    elif all(t_ok) and all(e["act_minus_tfidf_ci"][1] < 0.10 for e in transfer.values()):
        q2_verdict = "words_suffice"
    else:
        q2_verdict = "ambiguous"

    # q3: template-wrapped dolma vs the same text without the template, exp01 coarse bands
    q3 = {}
    for ood, ref in ORIGINAL:
        n_o, n_r = len(scores[ood]["length"]), len(scores[ref]["length"])
        boots = [(rng.integers(0, n_o, n_o), rng.integers(0, n_r, n_r)) for _ in range(config.n_boot)]
        for kind in ("dmaha", "dknn"):
            w, m = f"{kind}_wrapped", f"{kind}_matched"
            pw, pm = strat(w, ood, ref, bands=LENGTH_BANDS), strat(m, ood, ref, bands=LENGTH_BANDS)
            diffs = [(a - b) if a is not None and b is not None else None
                     for a, b in ((strat(w, ood, ref, i, j, bands=LENGTH_BANDS), strat(m, ood, ref, i, j, bands=LENGTH_BANDS)) for i, j in boots)]
            q3[f"{ood}_vs_{ref}|{kind}"] = {"wrapped": pw, "matched": pm, "full": strat(f"{kind}_full", ood, ref, bands=LENGTH_BANDS),
                                            "diff": None if pw is None or pm is None else pw - pm, "diff_ci": ci(diffs, 95)}
    q3_ok = all(e["diff_ci"] is not None for e in q3.values())
    if q3_ok and any(abs(e["diff"]) >= 0.05 and (e["diff_ci"][0] > 0 or e["diff_ci"][1] < 0) for e in q3.values()):
        q3_verdict = "template_mismatch_contributed"
    elif q3_ok and all(-0.05 <= e["diff_ci"][0] and e["diff_ci"][1] <= 0.05 for e in q3.values()):
        q3_verdict = "no_contribution"
    else:
        q3_verdict = "ambiguous"

    decision = {
        "cell": f"{config.pooling} L{config.layer}, dolma full, fine bands (q3: coarse bands)",
        "reproduction_check": repro,
        "q1": {"verdict": q1_verdict,
               "gates": {"G1_topic_kept": {"passed": g1, **topic_rates}, "G2_words_gone": {"passed": g2, "tfidf_stratified": g2_detail},
                         "G3_balance": {"passed": g3, **{k: e["length_only_stratified"] for k, e in primary.items()}}},
               "primary": {k: {d: e["auroc_stratified"][d] for d in (*BAR_DETECTORS, "tfidf")} | {f"delta_{d}": e[f"delta_{d}_minus_tfidf"] for d in BAR_DETECTORS}
                           | {f"delta_{d}_ci": e[f"delta_{d}_ci"] for d in BAR_DETECTORS} for k, e in primary.items()},
               "per_detector": {d: {"adds_beyond_words": adds(d), "words_only": words_only(d)} for d in BAR_DETECTORS}},
        "q2": {"verdict": q2_verdict, **{p: {k: transfer[p][k] for k in ("transfer_stratified", "act_minus_tfidf", "act_minus_tfidf_ci", "act_minus_minilm")} for p in transfer}},
        "q3": {"verdict": q3_verdict, **q3},
    }

    # exploratory: clean rows, tf-idf-stratified readout, q3 detectors on rewrites, other cells
    clean = {}
    for ood, ref in PRIMARY:
        i, j = np.where(on_topic[ood] & leak_free[ood])[0], np.where(on_topic[ref] & leak_free[ref])[0]
        clean[f"{ood}_vs_{ref}"] = {"n": [len(i), len(j)], **{d: strat(d, ood, ref, i, j) for d in (*BAR_DETECTORS, "tfidf", "length")}}
    tfidf_strat = {}
    for ood, ref in [*PRIMARY, *ORIGINAL]:
        pooled = np.r_[tfidf_scores[ood], tfidf_scores[ref]]
        edges = tuple(np.quantile(pooled, np.linspace(0, 1, 6))[:-1]) + (pooled.max() + 1e-9,)
        tfidf_strat[f"{ood}_vs_{ref}"] = {d: stratified_auroc(scores[ood][d], scores[ref][d], tfidf_scores[ood], tfidf_scores[ref], edges)
                                          for d in ("dmaha_full", "dknn_full", "length", "tfidf")}
    q3_on_rewrites = {f"{o}_vs_{r}": {d: strat(d, o, r) for d in ("dmaha_matched", "dmaha_wrapped", "dknn_matched", "dknn_wrapped")} for o, r in PRIMARY}
    original_fine = {f"{o}_vs_{r}": {d: strat(d, o, r) for d in DETECTORS} for o, r in ORIGINAL}
    original_coarse = {f"{o}_vs_{r}": {d: strat(d, o, r, bands=LENGTH_BANDS) for d in DETECTORS} for o, r in ORIGINAL}
    cells = {}
    for pooling in POOLINGS:
        for layer in LAYERS:
            sc = scores if (pooling, layer) == (config.pooling, config.layer) else score_cell(pooling, layer)
            cells[f"{pooling}|L{layer}"] = {f"{o}_vs_{r}": {d: strat(d, o, r, sc=sc) for d in DETECTORS[:8]} for o, r in [*PRIMARY, *ORIGINAL]}
            LOGGER.info(f"exploratory cell {pooling} L{layer} done")

    write_json(decision, run / "decision.json")
    write_json({"primary": primary, "transfer": transfer, "q3": q3, "original_fine_bands": original_fine, "original_coarse_bands": original_coarse},
               run / "metrics_primary.json")
    write_json({"clean_rows_only": clean, "tfidf_stratified": tfidf_strat, "q3_on_rewrites": q3_on_rewrites, "cells": cells}, run / "metrics_exploratory.json")
    out_rows = [{"id": str(rid), "pool": n, "n_content_tokens": int(lengths[n][k]), **{d: float(scores[n][d][k]) for d in DETECTORS}}
                for n in test_pools for k, rid in enumerate(feats[n]["ids"])]
    write_jsonl(out_rows, run / "scores_decision_cell.jsonl")
    print(f"q1 verdict: {q1_verdict}")
    print(f"q2 verdict: {q2_verdict}")
    print(f"q3 verdict: {q3_verdict}")
    print(f"decision written to {run / 'decision.json'}")


if __name__ == "__main__":
    main()
