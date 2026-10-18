# exp05: scores style-rewritten legal, medical, wildchat and idmix pools with the exp01 detectors and evaluates the bar.
# uv run -m scripts.run_rewrite_detectors --run_dir results/raw/261018_exp05_rewrite_v1 --exp01_dir results/raw/261018_exp01_repro_v1 --exp01_pools data/processed/exp01 --exp05_pools data/processed/exp05 --seed 0

import os

os.environ.setdefault("OMP_NUM_THREADS", "16")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "16")

import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import simple_parsing

from src.data.io import load_jsonl
from src.interp.features import LAYERS, POOLINGS
from src.metrics.detectors import KNN, Mahalanobis, auroc, stratified_auroc, tfidf_mix_vs_dolma
from src.metrics.io import write_json, write_jsonl
from src.utils.git import get_git_hash
from src.utils.logging import setup_logging
from src.utils.seed import set_seed

LOGGER = logging.getLogger(__name__)

# identical to exp04's bands: log-spaced by 2^(1/4) from 1 to 1024 content tokens
FINE_BANDS = tuple(float(b) for b in np.unique(np.round(2 ** (np.arange(0, 41) / 4)))) + (1025.0,)
EXP01_POOLS = ["idfit", "dolma_full", "dolma_matched", "legal", "medical", "wildchat_en_test", "idmix"]
RW_POOLS = ["legal_rw", "medical_rw", "wildchat_rw", "idmix_rw"]
BAR_DETECTORS = ("dmaha_full", "dknn_full")
DETECTORS = ("maha", "knn", "dmaha_full", "dknn_full", "dmaha_matched", "dknn_matched", "tfidf", "length")
PRIMARY = [("legal_rw", "wildchat_rw"), ("medical_rw", "wildchat_rw")]
ORIGINAL = [("legal", "wildchat_en_test"), ("medical", "wildchat_en_test")]
CONTEXT = [("idmix_rw", "idmix"), ("wildchat_rw", "wildchat_en_test"), ("legal_rw", "wildchat_en_test"), ("medical_rw", "wildchat_en_test"),
           ("legal_rw", "idmix_rw"), ("medical_rw", "idmix_rw")]


@dataclass
class RewriteDetectorConfig:
    run_dir: str = ""
    exp01_dir: str = ""
    exp01_pools: str = ""
    exp05_pools: str = ""
    seed: int = 0
    n_boot: int = 1000
    pooling: str = "cmean"
    layer: int = 16
    dolma_tfidf_words: int = 400
    ci_level: float = 98.75


def fit_detectors(x: dict[str, np.ndarray]):
    scale = x["idfit"].std(0) + 1e-6
    maha_mix, knn_mix = Mahalanobis(x["idfit"]), KNN(x["idfit"], scale)
    dolma = {v: (Mahalanobis(x[f"dolma_{v}"]), KNN(x[f"dolma_{v}"], scale)) for v in ("full", "matched")}

    def score(feats: np.ndarray) -> dict[str, np.ndarray]:
        m, k = maha_mix(feats), knn_mix(feats)
        out = {"maha": m, "knn": k}
        for v, (md, kd) in dolma.items():
            out[f"dmaha_{v}"], out[f"dknn_{v}"] = m - md(feats), k - kd(feats)
        return out

    return score


def main():
    config = simple_parsing.parse(RewriteDetectorConfig)
    run = Path(config.run_dir)
    setup_logging(run)
    set_seed(config.seed)
    write_json({"git_hash": get_git_hash(), **asdict(config), "fine_bands": FINE_BANDS}, run / "detector_config.json")

    feats = {n: dict(np.load(Path(config.exp01_dir) / "features" / f"{n}.npz")) for n in EXP01_POOLS}
    feats |= {n: dict(np.load(run / "features" / f"{n}.npz")) for n in RW_POOLS}
    rows = {n: load_jsonl(Path(config.exp01_pools) / f"{n}.jsonl") for n in ["idfit", "dolma", "legal", "medical", "wildchat_en_test"]}
    rows |= {n: load_jsonl(Path(config.exp05_pools) / f"{n}.jsonl") for n in RW_POOLS}
    rows["idmix"] = load_jsonl(Path(config.exp05_pools) / "idmix_sub.jsonl")
    # restrict exp01 idmix features to the 500 rows that were rewritten
    pos = {str(i): k for k, i in enumerate(feats["idmix"]["ids"])}
    sub = [pos[r["id"]] for r in rows["idmix"]]
    feats["idmix"] = {k: v[sub] for k, v in feats["idmix"].items()}
    for n in [*RW_POOLS, "legal", "medical", "wildchat_en_test", "idmix", "idfit"]:
        assert [str(i) for i in feats[n]["ids"]] == [r["id"] for r in rows[n]], f"{n} rows misaligned"
    texts = {n: [r["text"] for r in rs] for n, rs in rows.items()}
    lengths = {n: f["n_content_tokens"] for n, f in feats.items()}
    judge_ok = {n: np.array([r["judge_ok"] is True for r in rows[n]]) for n in RW_POOLS}

    tfidf = tfidf_mix_vs_dolma(texts["idfit"], [" ".join(t.split()[: config.dolma_tfidf_words]) for t in texts["dolma"]], config.seed)
    test_pools = ["legal", "medical", "wildchat_en_test", "idmix", *RW_POOLS]
    tfidf_scores = {n: tfidf(texts[n]) for n in test_pools}

    def score_cell(pooling: str, layer: int) -> dict[str, dict[str, np.ndarray]]:
        li = LAYERS.index(layer)
        x = {n: f[pooling][:, li].astype(np.float32) for n, f in feats.items()}
        score = fit_detectors(x)
        return {n: score(x[n]) | {"tfidf": tfidf_scores[n], "length": lengths[n].astype(float)} for n in test_pools}

    scores = score_cell(config.pooling, config.layer)
    LOGGER.info(f"scored decision cell {config.pooling} L{config.layer}")

    def strat(det: str, ood: str, ref: str, i=None, j=None, sc=None) -> float | None:
        sc = sc or scores
        o, r, lo, lr = sc[ood][det], sc[ref][det], lengths[ood], lengths[ref]
        if i is not None:
            o, lo, r, lr = o[i], lo[i], r[j], lr[j]
        return stratified_auroc(o, r, lo, lr, FINE_BANDS)["auroc"]

    def compare(det: str, ood: str, ref: str) -> dict:
        s = stratified_auroc(scores[ood][det], scores[ref][det], lengths[ood], lengths[ref], FINE_BANDS)
        return {"detector": det, "ood": ood, "ref": ref, "auroc_raw": auroc(scores[ood][det], scores[ref][det]),
                "auroc_stratified": s["auroc"], "n_bands": s["n_bands"], "coverage": s["coverage"]}

    lo_q, hi_q = (100 - config.ci_level) / 2, 100 - (100 - config.ci_level) / 2
    rng = np.random.default_rng(config.seed)
    primary = {}
    for ood, ref in PRIMARY:
        n_o, n_r = len(scores[ood]["length"]), len(scores[ref]["length"])
        boots = [(rng.integers(0, n_o, n_o), rng.integers(0, n_r, n_r)) for _ in range(config.n_boot)]
        point = {d: strat(d, ood, ref) for d in DETECTORS}
        boot_vals = {d: [strat(d, ood, ref, i, j) for i, j in boots] for d in (*BAR_DETECTORS, "tfidf")}
        entry = {"auroc_stratified": point, "auroc_raw": {d: auroc(scores[ood][d], scores[ref][d]) for d in DETECTORS},
                 "length_only_stratified": point["length"], "balanced": point["length"] is not None and 0.45 <= point["length"] <= 0.55}
        for d in BAR_DETECTORS:
            pairs = [(a, t) for a, t in zip(boot_vals[d], boot_vals["tfidf"]) if a is not None and t is not None]
            deltas = [a - t for a, t in pairs]
            entry[f"{d}_ci"] = [float(np.percentile([a for a, _ in pairs], q)) for q in (lo_q, hi_q)]
            entry[f"delta_{d}_minus_tfidf"] = None if point[d] is None or point["tfidf"] is None else point[d] - point["tfidf"]
            entry[f"delta_{d}_ci"] = [float(np.percentile(deltas, q)) for q in (lo_q, hi_q)] if len(deltas) >= 0.9 * config.n_boot else None
        entry["tfidf_ci"] = [float(np.percentile([v for v in boot_vals["tfidf"] if v is not None], q)) for q in (lo_q, hi_q)]
        primary[f"{ood}_vs_{ref}"] = entry
        LOGGER.info(f"{ood} vs {ref}: {entry}")

    original = {f"{o}_vs_{r}": {d: strat(d, o, r) for d in DETECTORS} for o, r in ORIGINAL}
    context = [compare(d, o, r) for o, r in CONTEXT for d in DETECTORS]

    yes_rates = {n: float(judge_ok[n].mean()) for n in RW_POOLS}
    g1 = all(v >= 0.90 for v in yes_rates.values())
    g2_detail = {p: {"tfidf_rewritten": primary[f"{p}_rw_vs_wildchat_rw"]["auroc_stratified"]["tfidf"],
                     "tfidf_original": original[f"{p}_vs_wildchat_en_test"]["tfidf"]} for p in ("legal", "medical")}
    g2 = all(v["tfidf_rewritten"] is not None and v["tfidf_original"] is not None and v["tfidf_rewritten"] <= v["tfidf_original"] - 0.10
             for v in g2_detail.values())
    g3 = all(e["balanced"] for e in primary.values())

    def adds(d: str) -> bool:
        return all(e["auroc_stratified"][d] is not None and e["auroc_stratified"][d] >= 0.75 and e[f"delta_{d}_minus_tfidf"] >= 0.10
                   and e[f"delta_{d}_ci"] is not None and e[f"delta_{d}_ci"][0] > 0 for e in primary.values())

    def words_only(d: str) -> bool:
        return all(e["auroc_stratified"][d] is not None and (e["auroc_stratified"][d] <= 0.65 or (e[f"delta_{d}_ci"] is not None and e[f"delta_{d}_ci"][1] < 0.10))
                   for e in primary.values())

    if not (g1 and g3):
        verdict = f"inconclusive: gate failed (G1 content {g1}, G3 balance {g3})"
    elif any(adds(d) for d in BAR_DETECTORS):
        verdict = "activations_add_beyond_words"
    elif all(words_only(d) for d in BAR_DETECTORS):
        verdict = "words_only" if g2 else "inconclusive: words-only pattern but G2 failed (rewrite did not remove the wording cue)"
    else:
        verdict = "ambiguous"
    decision = {"cell": f"{config.pooling} L{config.layer}, dolma full, fine bands", "verdict": verdict,
                "gates": {"G1_content": {"passed": g1, "judge_yes_rate": yes_rates}, "G2_wording_moved": {"passed": g2, **g2_detail},
                          "G3_balance": {"passed": g3, **{k: e["length_only_stratified"] for k, e in primary.items()}}},
                "primary": {k: {d: e["auroc_stratified"][d] for d in (*BAR_DETECTORS, "tfidf")} | {f"delta_{d}": e[f"delta_{d}_minus_tfidf"] for d in BAR_DETECTORS}
                            | {f"delta_{d}_ci": e[f"delta_{d}_ci"] for d in BAR_DETECTORS} for k, e in primary.items()},
                "per_detector": {d: {"adds_beyond_words": adds(d), "words_only": words_only(d)} for d in BAR_DETECTORS}}

    # exploratory: judge-approved rows only, and every layer and pooling for the primary comparisons
    judged_only = {}
    for ood, ref in PRIMARY:
        i, j = np.where(judge_ok[ood])[0], np.where(judge_ok[ref])[0]
        judged_only[f"{ood}_vs_{ref}"] = {d: strat(d, ood, ref, i, j) for d in (*BAR_DETECTORS, "tfidf", "length")}
    cells = {}
    for pooling in POOLINGS:
        for layer in LAYERS:
            sc = scores if (pooling, layer) == (config.pooling, config.layer) else score_cell(pooling, layer)
            cells[f"{pooling}|L{layer}"] = {f"{o}_vs_{r}": {d: strat(d, o, r, sc=sc) for d in DETECTORS[:6]} for o, r in [*PRIMARY, *ORIGINAL]}
            LOGGER.info(f"exploratory cell {pooling} L{layer} done")

    write_json(decision, run / "decision.json")
    write_json({"primary": primary, "original_fine_bands": original, "context": context}, run / "metrics_primary.json")
    write_json({"judge_ok_only": judged_only, "cells": cells}, run / "metrics_exploratory.json")
    out_rows = [{"id": str(rid), "pool": n, "n_content_tokens": int(lengths[n][k]), **{d: float(scores[n][d][k]) for d in DETECTORS}}
                for n in test_pools for k, rid in enumerate(feats[n]["ids"])]
    write_jsonl(out_rows, run / "scores_decision_cell.jsonl")
    print(f"verdict: {verdict}")
    print(f"decision written to {run / 'decision.json'}")


if __name__ == "__main__":
    main()
