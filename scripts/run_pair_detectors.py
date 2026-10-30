# repro_olmo2_1b: fits the repro_two_sided detectors on a second model pair's cached features and evaluates the second-pair bars.
# uv run --no-sync -m scripts.run_pair_detectors --run_dir results/raw/261019_repro_olmo2_1b_v1 --pools_dir data/processed/repro_olmo2_1b --decision_layer 8 --seed 0

import os

os.environ.setdefault("OMP_NUM_THREADS", "16")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "16")

import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import simple_parsing
from scipy.stats import spearmanr

from src.data.io import load_jsonl
from src.interp.features import POOLINGS
from src.metrics.detectors import evaluate_pair, score_cell, tfidf_mix_vs_dolma, with_ci
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
ONE_SIDED = {"dmaha_full": "maha", "dknn_full": "knn"}
OOD = ("legal", "medical")


@dataclass
class PairDetectorConfig:
    run_dir: str = ""
    features_dir: str = ""
    pools_dir: str = ""
    decision_layer: int = 8
    seed: int = 0
    n_boot: int = 1000
    dolma_tfidf_words: int = 400


def main():
    config = simple_parsing.parse(PairDetectorConfig)
    assert config.run_dir and config.pools_dir, "run_dir and pools_dir are required"
    run = Path(config.run_dir)
    features_dir = Path(config.features_dir) if config.features_dir else run / "features"
    setup_logging(run)
    set_seed(config.seed)
    write_json({"git_hash": get_git_hash(), **asdict(config)}, run / "detector_config.json")
    names = (*TEST_POOLS, "idfit", "dolma_full", "dolma_matched")
    feats = {name: dict(np.load(features_dir / f"{name}.npz")) for name in names}
    layers = [int(l) for l in feats["idfit"]["layers"]]
    lengths = {name: feats[name]["n_content_tokens"] for name in names}
    texts = {
        name: [r["text"] for r in load_jsonl(Path(config.pools_dir) / f"{name}.jsonl")]
        for name in (*TEST_POOLS, "idfit", "dolma")
    }
    for name in names:
        pool_file = name.replace("_full", "").replace("_matched", "")
        expected = [r["id"] for r in load_jsonl(Path(config.pools_dir) / f"{pool_file}.jsonl")]
        assert list(feats[name]["ids"]) == expected, f"{name} feature rows do not line up with the pool file"

    baselines = {pool: {"length": lengths[pool].astype(float)} for pool in TEST_POOLS}
    tfidf = tfidf_mix_vs_dolma(
        texts["idfit"], [" ".join(t.split()[: config.dolma_tfidf_words]) for t in texts["dolma"]], config.seed
    )
    for pool in TEST_POOLS:
        baselines[pool]["tfidf"] = tfidf(texts[pool])

    rows, decision_scores = [], None
    for pooling in POOLINGS:
        for li, layer in enumerate(layers):
            x = {name: feats[name][pooling][:, li].astype(np.float32) for name in names}
            scores = score_cell(x, TEST_POOLS)
            rows += [
                {"pooling": pooling, "layer": layer, **evaluate_pair(scores, lengths, d, o, r)}
                for d in scores["idmix"]
                for o, r in COMPARISONS
            ]
            if (pooling, layer) == ("cmean", config.decision_layer):
                decision_scores = scores
            LOGGER.info(f"{pooling} L{layer} done")
    assert decision_scores is not None, f"decision layer {config.decision_layer} not in {layers}"

    cell_rows = [
        with_ci(
            {"pooling": "cmean", "layer": config.decision_layer, **evaluate_pair(decision_scores, lengths, d, o, r)},
            decision_scores,
            lengths,
            config.n_boot,
            config.seed,
        )
        for d in decision_scores["idmix"]
        for o, r in COMPARISONS
    ]
    base_rows = [
        with_ci(
            {"pooling": None, "layer": None, **evaluate_pair(baselines, lengths, d, o, r)},
            baselines,
            lengths,
            config.n_boot,
            config.seed,
        )
        for d in ("length", "tfidf")
        for o, r in COMPARISONS
    ]

    def get(detector: str, ood: str, ref: str, key: str):
        return next(
            r[key] for r in cell_rows + base_rows if (r["detector"], r["ood"], r["ref"]) == (detector, ood, ref)
        )

    positive = {p: get("knn", p, "idmix", "auroc_raw") for p in OOD}
    is_positive = all(v >= 0.90 for v in positive.values())
    length_only = {p: get("length", p, "wildchat_en_test", "auroc_stratified") for p in OOD}
    is_measurable = all(v is not None and 0.45 <= v <= 0.55 for v in length_only.values())
    strat = {
        d: {p: get(d, p, "wildchat_en_test", "auroc_stratified") for p in OOD}
        for d in ("maha", "knn", "dmaha_full", "dknn_full", "tfidf")
    }
    has_values = all(v is not None for d in strat.values() for v in d.values())
    q1 = q2 = None
    if is_positive and is_measurable and has_values:
        reproduces = all(strat[o][p] <= 0.60 for o in ("maha", "knn") for p in OOD) and all(
            strat[t][p] >= 0.85 for t in ONE_SIDED for p in OOD
        )
        no_advantage = all(strat[t][p] - strat[o][p] <= 0.05 for t, o in ONE_SIDED.items() for p in OOD)
        q1 = "pattern_reproduces" if reproduces else ("no_two_sided_advantage" if no_advantage else "ambiguous")
        best = {p: max(strat["dmaha_full"][p], strat["dknn_full"][p]) for p in OOD}
        if all(strat["tfidf"][p] >= best[p] - 0.05 for p in OOD):
            q2 = "lexical_parity"
        elif all(strat["tfidf"][p] <= best[p] - 0.10 for p in OOD):
            q2 = "activations_beat_words"
        else:
            q2 = "ambiguous"
    length_rho = {p: float(spearmanr(decision_scores[p]["maha"], lengths[p]).statistic) for p in TEST_POOLS}
    decision = {
        "cell": f"cmean L{config.decision_layer}, dolma full, layers {layers}",
        "positive_control": {"knn_vs_idmix_raw": positive, "threshold": ">= 0.90", "cleared": is_positive},
        "measurability": {"length_only_stratified": length_only, "band": "[0.45, 0.55]", "measurable": is_measurable},
        "stratified_vs_wildchat": strat,
        "q1_second_pair": {
            "verdict": q1,
            "reproduces_rule": "one-sided <= 0.60 and two-sided >= 0.85 on both pools",
            "no_advantage_rule": "two-sided minus one-sided <= 0.05 on both pools",
        },
        "q2_lexical": {"verdict": q2},
        "near_id_control_wildchat_vs_idmix": {
            d: get(d, "wildchat_en_test", "idmix", "auroc_stratified")
            for d in ("maha", "knn", "dmaha_full", "dknn_full", "tfidf")
        },
        "maha_spearman_with_length": length_rho,
    }
    write_json(rows, run / "metrics_all_cells.json")
    write_json({"decision_cell": cell_rows, "baselines": base_rows}, run / "metrics_decision_cell.json")
    write_json(decision, run / "decision.json")
    write_jsonl(
        [
            {
                "id": str(feats[pool]["ids"][i]),
                "pool": pool,
                "n_content_tokens": int(lengths[pool][i]),
                **{d: float(decision_scores[pool][d][i]) for d in decision_scores[pool]},
                **{d: float(baselines[pool][d][i]) for d in baselines[pool]},
            }
            for pool in TEST_POOLS
            for i in range(len(feats[pool]["ids"]))
        ],
        run / "scores_decision_cell.jsonl",
    )
    print(f"decision written to {run / 'decision.json'}")


if __name__ == "__main__":
    main()
