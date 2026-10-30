# scores register_vs_coverage reddit pools with the repro_two_sided detectors (cmean L16) and evaluates the register-vs-coverage bar.
# uv run -m scripts.run_reddit_detectors --run_dir results/raw/261018_register_vs_coverage_reddit_v1 --repro_two_sided_dir results/raw/261018_repro_two_sided_repro_v1 --repro_two_sided_pools data/processed/repro_two_sided --register_vs_coverage_pools data/processed/register_vs_coverage --seed 0

import os

os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "8")

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import simple_parsing

from src.data.io import load_jsonl
from src.interp.features import LAYERS
from src.metrics.detectors import KNN, Mahalanobis, auroc, bootstrap_ci, stratified_auroc, tfidf_mix_vs_dolma
from src.metrics.io import write_json, write_jsonl
from src.utils.logging import setup_logging
from src.utils.seed import set_seed

LOGGER = logging.getLogger(__name__)

COVERED = [
    "reddit_relationship_advice",
    "reddit_fitness",
    "reddit_cooking",
    "reddit_writing",
    "reddit_relationships",
    "reddit_learnprogramming",
    "reddit_askscience",
]
OOD = ["legal_rtb", "legal"]
EXP01_TEST = ["idmix", "wildchat_en_test", "legal"]
BAR_DETECTORS = ("dmaha_full", "dknn_full")
DETECTORS = ("maha", "knn", "dmaha_full", "dknn_full", "dmaha_matched", "dknn_matched", "tfidf", "length")


@dataclass
class RedditDetectorConfig:
    run_dir: str = ""
    repro_two_sided_dir: str = ""
    repro_two_sided_pools: str = ""
    register_vs_coverage_pools: str = ""
    seed: int = 0
    n_boot: int = 1000
    pooling: str = "cmean"
    layer: int = 16
    dolma_tfidf_words: int = 400


def main():
    config = simple_parsing.parse(RedditDetectorConfig)
    run = Path(config.run_dir)
    setup_logging(run)
    set_seed(config.seed)
    li = LAYERS.index(config.layer)
    repro_two_sided_feats = {
        n: np.load(Path(config.repro_two_sided_dir) / "features" / f"{n}.npz")
        for n in ["idfit", "dolma_full", "dolma_matched", *EXP01_TEST]
    }
    register_vs_coverage_feats = {n: np.load(run / "features" / f"{n}.npz") for n in [*COVERED, "legal_rtb"]}
    feats = {**repro_two_sided_feats, **register_vs_coverage_feats}
    texts = {
        n: [r["text"] for r in load_jsonl(Path(config.repro_two_sided_pools) / f"{n}.jsonl")]
        for n in ["idfit", "dolma", *EXP01_TEST]
    }
    texts |= {
        n: [r["text"] for r in load_jsonl(Path(config.register_vs_coverage_pools) / f"{n}.jsonl")]
        for n in [*COVERED, "legal_rtb"]
    }
    for n in register_vs_coverage_feats:
        assert list(feats[n]["ids"]) == [
            r["id"] for r in load_jsonl(Path(config.register_vs_coverage_pools) / f"{n}.jsonl")
        ], f"{n} rows misaligned"

    x = {n: f[config.pooling][:, li].astype(np.float32) for n, f in feats.items()}
    lengths = {n: f["n_content_tokens"] for n, f in feats.items()}
    scale = x["idfit"].std(0) + 1e-6
    maha_mix, knn_mix = Mahalanobis(x["idfit"]), KNN(x["idfit"], scale)
    dolma = {v: (Mahalanobis(x[f"dolma_{v}"]), KNN(x[f"dolma_{v}"], scale)) for v in ("full", "matched")}
    tfidf = tfidf_mix_vs_dolma(
        texts["idfit"], [" ".join(t.split()[: config.dolma_tfidf_words]) for t in texts["dolma"]], config.seed
    )

    test_pools = [*EXP01_TEST, *COVERED, "legal_rtb"]
    scores = {}
    for n in test_pools:
        m, k = maha_mix(x[n]), knn_mix(x[n])
        scores[n] = {"maha": m, "knn": k, "tfidf": tfidf(texts[n]), "length": lengths[n].astype(float)}
        for v, (md, kd) in dolma.items():
            scores[n][f"dmaha_{v}"] = m - md(x[n])
            scores[n][f"dknn_{v}"] = k - kd(x[n])
        LOGGER.info(f"scored {n}")

    def compare(det: str, ood: str, ref: str, with_ci: bool) -> dict:
        o, r, lo, lr = scores[ood][det], scores[ref][det], lengths[ood], lengths[ref]
        strat = stratified_auroc(o, r, lo, lr)
        out = {
            "detector": det,
            "ood": ood,
            "ref": ref,
            "auroc_raw": auroc(o, r),
            "auroc_stratified": strat["auroc"],
            "strat_n_bands": strat["n_bands"],
            "strat_coverage": strat["coverage"],
        }
        if with_ci:
            out["ci_stratified"] = bootstrap_ci(
                lambda i, j: stratified_auroc(o[i], r[j], lo[i], lr[j])["auroc"],
                len(o),
                len(r),
                config.n_boot,
                config.seed,
            )
        return out

    primary = [
        compare(d, ood, ref, d in BAR_DETECTORS and ood == "legal_rtb")
        for d in DETECTORS
        for ood in OOD
        for ref in COVERED
    ]
    context = [
        compare(d, n, "wildchat_en_test", False) | {"auroc_vs_idmix_raw": auroc(scores[n][d], scores["idmix"][d])}
        for d in DETECTORS
        for n in [*COVERED, "legal_rtb", "legal"]
    ]

    summary = {}
    for d in DETECTORS:
        for ood in OOD:
            vals = [r["auroc_stratified"] for r in primary if r["detector"] == d and r["ood"] == ood]
            usable = [v for v in vals if v is not None]
            summary[f"{d}|{ood}"] = {
                "median_over_subreddits": float(np.median(usable)) if usable else None,
                "n_measurable": len(usable),
                "min": float(min(usable)) if usable else None,
                "max": float(max(usable)) if usable else None,
            }

    def verdict(med: float | None) -> str:
        if med is None:
            return "unmeasurable"
        return "separates_coverage" if med >= 0.75 else ("reads_register" if med <= 0.60 else "ambiguous")

    decision = {
        "metric": "length-stratified AUROC, legal_rtb (ood) vs each covered subreddit, median over 7",
        "cell": f"{config.pooling} L{config.layer}",
        **{
            d: {
                "median": summary[f"{d}|legal_rtb"]["median_over_subreddits"],
                "verdict": verdict(summary[f"{d}|legal_rtb"]["median_over_subreddits"]),
            }
            for d in BAR_DETECTORS
        },
    }
    write_json({"primary": primary, "summary": summary}, run / "metrics_primary.json")
    write_json(context, run / "metrics_context.json")
    write_json(decision, run / "decision.json")
    rows = []
    for n in test_pools:
        for i, rid in enumerate(feats[n]["ids"]):
            rows.append(
                {
                    "id": str(rid),
                    "pool": n,
                    "n_content_tokens": int(lengths[n][i]),
                    **{d: float(scores[n][d][i]) for d in DETECTORS},
                }
            )
    write_jsonl(rows, run / "scores_decision_cell.jsonl")
    print(f"decision written to {run / 'decision.json'}")


if __name__ == "__main__":
    main()
