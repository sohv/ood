# fits one- and two-sided ood detectors on cached repro_two_sided features and evaluates the pre-registered bars.
# uv run -m scripts.run_detectors --run_dir results/raw/261018_repro_pca_maha_v1 --features_dir results/raw/261018_repro_two_sided_repro_v1/features --pools_dir data/processed/repro_two_sided --maha_pca_dims 256 --evaluate_bar2_always --seed 0

import os

os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "8")

import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import simple_parsing

from src.data.io import load_jsonl
from src.interp.features import LAYERS, POOLINGS
from src.metrics.detectors import KNN, Mahalanobis, auroc, bootstrap_ci, stratified_auroc, tfidf_mix_vs_dolma
from src.metrics.io import write_json, write_jsonl
from src.utils.git import get_git_hash
from src.utils.logging import setup_logging
from src.utils.seed import set_seed

LOGGER = logging.getLogger(__name__)

TEST_POOLS = ("idmix", "legal", "medical", "wildchat_en_test")
COMPARISONS = [
    ("legal", "idmix"),
    ("medical", "idmix"),
    ("wildchat_en_test", "idmix"),
    ("legal", "wildchat_en_test"),
    ("medical", "wildchat_en_test"),
]
DECISION_CELL = ("cmean", 16)
ONE_SIDED = ("maha", "knn")
TWO_SIDED_BAR = ("dmaha_full", "dknn_full")


@dataclass
class DetectorConfig:
    run_dir: str = ""
    features_dir: str = ""
    pools_dir: str = ""
    maha_pca_dims: int = 0
    evaluate_bar2_always: bool = False
    exploratory_maha: bool = False
    seed: int = 0
    n_boot: int = 1000
    dolma_tfidf_words: int = 400


def cell_scores(feats: dict, pooling: str, layer_idx: int, **maha_kwargs) -> dict[str, dict[str, np.ndarray]]:
    x = {name: f[pooling][:, layer_idx].astype(np.float32) for name, f in feats.items()}
    scale = x["idfit"].std(0) + 1e-6
    maha_mix, knn_mix = Mahalanobis(x["idfit"], **maha_kwargs), KNN(x["idfit"], scale)
    refs = {v: (Mahalanobis(x[f"dolma_{v}"], **maha_kwargs), KNN(x[f"dolma_{v}"], scale)) for v in ("full", "matched")}
    scores = {}
    for pool in TEST_POOLS:
        m, k = maha_mix(x[pool]), knn_mix(x[pool])
        scores[pool] = {"maha": m, "knn": k}
        for v, (maha_d, knn_d) in refs.items():
            scores[pool][f"dmaha_{v}"] = m - maha_d(x[pool])
            scores[pool][f"dknn_{v}"] = k - knn_d(x[pool])
    return scores


def evaluate(scores: dict, lengths: dict, detector: str, ood: str, ref: str) -> dict:
    o, r = scores[ood][detector], scores[ref][detector]
    strat = stratified_auroc(o, r, lengths[ood], lengths[ref])
    return {
        "detector": detector,
        "ood": ood,
        "ref": ref,
        "auroc_raw": auroc(o, r),
        "auroc_stratified": strat["auroc"],
        "strat_n_bands": strat["n_bands"],
        "strat_coverage": strat["coverage"],
        "strat_bands": strat["bands"],
    }


def with_ci(row: dict, scores: dict, lengths: dict, n_boot: int, seed: int) -> dict:
    o, r = scores[row["ood"]][row["detector"]], scores[row["ref"]][row["detector"]]
    lo, lr = lengths[row["ood"]], lengths[row["ref"]]
    row["ci_raw"] = bootstrap_ci(lambda i, j: auroc(o[i], r[j]), len(o), len(r), n_boot, seed)
    row["ci_stratified"] = bootstrap_ci(
        lambda i, j: stratified_auroc(o[i], r[j], lo[i], lr[j])["auroc"], len(o), len(r), n_boot, seed
    )
    return row


def main():
    config = simple_parsing.parse(DetectorConfig)
    run = Path(config.run_dir)
    features_dir = Path(config.features_dir) if config.features_dir else run / "features"
    setup_logging(run)
    set_seed(config.seed)
    if not (run / "config.json").exists():
        write_json({"git_hash": get_git_hash(), **asdict(config)}, run / "config.json")
    names = (*TEST_POOLS, "idfit", "dolma_full", "dolma_matched")
    feats = {name: dict(np.load(features_dir / f"{name}.npz")) for name in names}
    lengths = {name: feats[name]["n_content_tokens"] for name in names}
    texts = {
        name: [r["text"] for r in load_jsonl(Path(config.pools_dir) / f"{name}.jsonl")]
        for name in (*TEST_POOLS, "idfit", "dolma")
    }
    for name in names:
        expected = [
            r["id"]
            for r in load_jsonl(Path(config.pools_dir) / f"{name.replace('_full', '').replace('_matched', '')}.jsonl")
        ]
        assert list(feats[name]["ids"]) == expected, f"{name} feature rows do not line up with the pool file"

    baselines = {pool: {"length": lengths[pool].astype(float)} for pool in TEST_POOLS}
    tfidf = tfidf_mix_vs_dolma(
        texts["idfit"], [" ".join(t.split()[: config.dolma_tfidf_words]) for t in texts["dolma"]], config.seed
    )
    for pool in TEST_POOLS:
        baselines[pool]["tfidf"] = tfidf(texts[pool])

    rows = []
    decision_scores = None
    for pooling in POOLINGS:
        for li, layer in enumerate(LAYERS):
            scores = cell_scores(feats, pooling, li, pca_dims=config.maha_pca_dims)
            for detector in scores["idmix"]:
                for ood, ref in COMPARISONS:
                    rows.append({"pooling": pooling, "layer": layer, **evaluate(scores, lengths, detector, ood, ref)})
            if (pooling, layer) == DECISION_CELL:
                decision_scores = scores
            LOGGER.info(f"{pooling} L{layer} done")

    decision_rows = [
        with_ci(
            {"pooling": "cmean", "layer": 16, **evaluate(decision_scores, lengths, d, o, r)},
            decision_scores,
            lengths,
            config.n_boot,
            config.seed,
        )
        for d in decision_scores["idmix"]
        for o, r in COMPARISONS
    ]
    baseline_rows = [
        with_ci(
            {"pooling": None, "layer": None, **evaluate(baselines, lengths, d, o, r)},
            baselines,
            lengths,
            config.n_boot,
            config.seed,
        )
        for d in ("length", "tfidf")
        for o, r in COMPARISONS
    ]

    def get(detector: str, ood: str, ref: str, key: str):
        return next(r[key] for r in decision_rows if (r["detector"], r["ood"], r["ref"]) == (detector, ood, ref))

    bar1 = {f"{d}|{p}_vs_idmix": get(d, p, "idmix", "auroc_raw") for d in ONE_SIDED for p in ("legal", "medical")}
    bar2_one = {
        f"{d}|{p}_vs_wildchat": get(d, p, "wildchat_en_test", "auroc_stratified")
        for d in ONE_SIDED
        for p in ("legal", "medical")
    }
    bar2_two = {
        f"{d}|{p}_vs_wildchat": get(d, p, "wildchat_en_test", "auroc_stratified")
        for d in TWO_SIDED_BAR
        for p in ("legal", "medical")
    }
    is_bar1 = all(v >= 0.95 for v in bar1.values())
    measurable = all(v is not None for v in (*bar2_one.values(), *bar2_two.values()))
    is_bar2_evaluated = is_bar1 or config.evaluate_bar2_always
    is_bar2 = measurable and all(v <= 0.60 for v in bar2_one.values()) and all(v >= 0.85 for v in bar2_two.values())
    decision = {
        "cell": "cmean L16, dolma full",
        "maha_pca_dims": config.maha_pca_dims,
        "bar1_positive_control": {"values": bar1, "threshold": ">= 0.95", "cleared": is_bar1},
        "bar2_main_claim": {
            "one_sided_stratified": bar2_one,
            "one_sided_threshold": "<= 0.60",
            "two_sided_stratified": bar2_two,
            "two_sided_threshold": ">= 0.85",
            "measurable": measurable,
            "evaluated": is_bar2_evaluated,
            "verdict": None
            if not is_bar2_evaluated
            else ("reproduced" if is_bar2 else ("ambiguous_unmeasurable" if not measurable else "not_reproduced")),
        },
    }

    write_json(rows, run / "metrics_all_cells.json")
    write_json({"decision_cell": decision_rows, "baselines": baseline_rows}, run / "metrics_decision_cell.json")
    write_json(decision, run / "decision.json")
    score_rows = []
    for pool in TEST_POOLS:
        ids = feats[pool]["ids"]
        for i in range(len(ids)):
            score_rows.append(
                {
                    "id": str(ids[i]),
                    "pool": pool,
                    "n_content_tokens": int(lengths[pool][i]),
                    **{d: float(decision_scores[pool][d][i]) for d in decision_scores[pool]},
                    **{d: float(baselines[pool][d][i]) for d in baselines[pool]},
                }
            )
    write_jsonl(score_rows, run / "scores_decision_cell.jsonl")

    if config.exploratory_maha:
        li = LAYERS.index(DECISION_CELL[1])
        variants = {
            **{f"pca{k}": {"pca_dims": k} for k in (32, 64, 128, 512, 1024)},
            **{f"shrink{a}": {"shrinkage": a} for a in (0.1, 0.5)},
        }
        explore = []
        for name, kwargs in variants.items():
            scores = cell_scores(feats, DECISION_CELL[0], li, **kwargs)
            for d in ("maha", "dmaha_full", "dmaha_matched"):
                explore += [{"variant": name, **evaluate(scores, lengths, d, o, r)} for o, r in COMPARISONS]
            explore.append(
                {
                    "variant": name,
                    "detector": "maha",
                    "ood": "idmix",
                    "ref": "idfit_in_sample",
                    "auroc_raw": auroc(
                        scores["idmix"]["maha"],
                        Mahalanobis(feats["idfit"][DECISION_CELL[0]][:, li], **kwargs)(
                            feats["idfit"][DECISION_CELL[0]][:, li]
                        ),
                    ),
                }
            )
            LOGGER.info(f"exploratory {name} done")
        write_json(explore, run / "metrics_exploratory_maha.json")
    print(f"decision written to {run / 'decision.json'}")
    print(f"all cells in {run / 'metrics_all_cells.json'}")


if __name__ == "__main__":
    main()
