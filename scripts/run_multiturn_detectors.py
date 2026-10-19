# exp13: scores multi-turn and tool-use pools against first-turn wildchat with two id references and evaluates the pre-registered bars.
# uv run --no-sync -m scripts.run_multiturn_detectors --run_dir results/raw/261019_exp13_multiturn_v1 --pools_dir data/processed/exp13 --exp01_dir results/raw/261018_exp01_repro_v1 --exp01_pools data/processed/exp01 --seed 0

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
from src.interp.features import LAYERS
from src.metrics.detectors import KNN, Mahalanobis, auroc, bootstrap_ci, stratified_auroc, tfidf_mix_vs_dolma
from src.metrics.io import write_json, write_jsonl
from src.utils.git import get_git_hash
from src.utils.logging import setup_logging
from src.utils.seed import set_seed

LOGGER = logging.getLogger(__name__)

NEW_POOLS = (
    "wc_mt_ctx",
    "wc_mt_alone",
    "dolci_mt_fit",
    "dolci_mt_test",
    "dolci_mt_test_alone",
    "tool_ctx",
    "tool_alone",
    "wildchat_en_test",
)
TEST_POOLS = (
    "wc_mt_ctx",
    "wc_mt_alone",
    "dolci_mt_test",
    "dolci_mt_test_alone",
    "tool_ctx",
    "tool_alone",
    "wildchat_en_test",
    "idmix",
)
REFS = ("wildchat_en_test", "idmix")
POOLINGS = ("cmean", "clast")
CELL = ("cmean", 16)
BAR_DETECTORS = ("dknn_full", "dmaha_full")
PAIRS = (("wc_mt_ctx", "wc_mt_alone"), ("dolci_mt_test", "dolci_mt_test_alone"), ("tool_ctx", "tool_alone"))


@dataclass
class MultiturnDetectorConfig:
    run_dir: str = ""
    features_dir: str = ""
    pools_dir: str = ""
    exp01_dir: str = ""
    exp01_pools: str = ""
    seed: int = 0
    n_boot: int = 1000
    dolma_tfidf_words: int = 400


def strat_ci(
    o: np.ndarray, r: np.ndarray, lo: np.ndarray, lr: np.ndarray, n_boot: int, seed: int
) -> list[float] | None:
    return bootstrap_ci(lambda i, j: stratified_auroc(o[i], r[j], lo[i], lr[j])["auroc"], len(o), len(r), n_boot, seed)


def paired_stat(c: np.ndarray, a: np.ndarray, w: np.ndarray, lc: np.ndarray, la: np.ndarray, lw: np.ndarray):
    # the same resampled rows score the context and alone versions, so the difference is paired
    def stat(i: np.ndarray, j: np.ndarray) -> float | None:
        p = stratified_auroc(c[i], w[j], lc[i], lw[j])["auroc"]
        q = stratified_auroc(a[i], w[j], la[i], lw[j])["auroc"]
        return None if p is None or q is None else p - q

    return stat


def band(value: float | None) -> str:
    if value is None:
        return "unmeasurable"
    return "high" if value >= 0.85 else ("low" if value <= 0.65 else "mid")


def main():
    config = simple_parsing.parse(MultiturnDetectorConfig)
    assert config.run_dir and config.pools_dir and config.exp01_dir and config.exp01_pools, "paths are required"
    run = Path(config.run_dir)
    features_dir = Path(config.features_dir) if config.features_dir else run / "features"
    setup_logging(run)
    set_seed(config.seed)
    write_json({"git_hash": get_git_hash(), **asdict(config)}, run / "detector_config.json")

    def load(feat_dir: Path, name: str, pool_dir: Path) -> dict:
        f = dict(np.load(feat_dir / f"{name}.npz"))
        rows = load_jsonl(pool_dir / f"{name.replace('_full', '')}.jsonl")
        assert list(f["ids"]) == [r["id"] for r in rows], f"{name} rows misaligned"
        f["text"] = [r["text"] for r in rows]
        return f

    e1f, e1p = Path(config.exp01_dir) / "features", Path(config.exp01_pools)
    feats = {name: load(e1f, name, e1p) for name in ("idfit", "idmix", "dolma_full")}
    feats |= {name: load(features_dir, name, Path(config.pools_dir)) for name in NEW_POOLS}
    lengths = {name: f["n_content_tokens"] for name, f in feats.items()}
    for i, r in enumerate(load_jsonl(Path(config.pools_dir) / "wc_mt_ctx.jsonl")[:5]):
        LOGGER.info(f"wc_mt_ctx raw example {i}: U1 {r['messages'][0]['content'][:200]!r} | U2 {r['text'][:200]!r}")

    baselines = {pool: {"length": lengths[pool].astype(float)} for pool in TEST_POOLS}
    tfidf = tfidf_mix_vs_dolma(
        feats["idfit"]["text"],
        [" ".join(t.split()[: config.dolma_tfidf_words]) for t in feats["dolma_full"]["text"]],
        config.seed,
    )
    for pool in TEST_POOLS:
        baselines[pool]["tfidf"] = tfidf(feats[pool]["text"])

    def cell(pooling: str, li: int, ref_name: str) -> dict[str, dict[str, np.ndarray]]:
        x = {name: f[pooling][:, li].astype(np.float32) for name, f in feats.items()}
        ref = x["idfit"] if ref_name == "idfit" else np.vstack([x["idfit"], x["dolci_mt_fit"]])
        scale = ref.std(0) + 1e-6
        maha_mix, knn_mix = Mahalanobis(ref), KNN(ref, scale)
        maha_d, knn_d = Mahalanobis(x["dolma_full"]), KNN(x["dolma_full"], scale)
        out = {}
        for pool in TEST_POOLS:
            m, k = maha_mix(x[pool]), knn_mix(x[pool])
            out[pool] = {"maha": m, "knn": k, "dmaha_full": m - maha_d(x[pool]), "dknn_full": k - knn_d(x[pool])}
        return out

    def evaluate(scores: dict, detector: str, ood: str, ref: str) -> dict:
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

    rows, cells = [], {}
    for ref_name in ("idfit", "idfit_mt"):
        for pooling in POOLINGS:
            for li, layer in enumerate(LAYERS):
                scores = cell(pooling, li, ref_name)
                cells[(ref_name, pooling, layer)] = scores
                rows += [
                    {"id_ref": ref_name, "pooling": pooling, "layer": layer, **evaluate(scores, d, o, r)}
                    for d in scores["idmix"]
                    for o in TEST_POOLS
                    for r in REFS
                    if o != r
                ]
            LOGGER.info(f"{ref_name} {pooling} done")
    write_json(rows, run / "metrics_all_cells.json")

    # pipeline check: re-extracted wildchat_en_test must reproduce exp01's stored decision-cell scores
    exp01_scores = {
        r["id"]: r
        for r in load_jsonl(Path(config.exp01_dir) / "scores_decision_cell.jsonl")
        if r["pool"] == "wildchat_en_test"
    }
    ids = [str(i) for i in feats["wildchat_en_test"]["ids"]]
    base = cells[("idfit", *CELL)]["wildchat_en_test"]
    pipeline = {
        d: float(spearmanr(base[d], [exp01_scores[i][d] for i in ids]).statistic) for d in ("dknn_full", "maha")
    }
    is_pipeline_ok = all(v >= 0.99 for v in pipeline.values())
    LOGGER.info(f"pipeline check {pipeline}")

    main_scores = cells[("idfit_mt", *CELL)]
    bar_rows = []
    for d in (*BAR_DETECTORS, "maha", "knn"):
        for o in TEST_POOLS:
            for r in REFS:
                if o != r:
                    row = {"id_ref": "idfit_mt", "pooling": CELL[0], "layer": CELL[1], **evaluate(main_scores, d, o, r)}
                    row["ci_stratified"] = strat_ci(
                        main_scores[o][d], main_scores[r][d], lengths[o], lengths[r], config.n_boot, config.seed
                    )
                    bar_rows.append(row)
    for d in ("length", "tfidf"):
        for o in TEST_POOLS:
            for r in REFS:
                if o != r:
                    bar_rows.append({"id_ref": None, "pooling": None, "layer": None, **evaluate(baselines, d, o, r)})
    paired = []
    for ctx, alone in PAIRS:
        for d in BAR_DETECTORS:
            stat = paired_stat(
                main_scores[ctx][d],
                main_scores[alone][d],
                main_scores["wildchat_en_test"][d],
                lengths[ctx],
                lengths[alone],
                lengths["wildchat_en_test"],
            )
            point = stat(np.arange(len(main_scores[ctx][d])), np.arange(len(main_scores["wildchat_en_test"][d])))
            paired.append(
                {
                    "pair": f"{ctx}_minus_{alone}",
                    "detector": d,
                    "difference": point,
                    "ci": bootstrap_ci(
                        stat, len(lengths[ctx]), len(lengths["wildchat_en_test"]), config.n_boot, config.seed
                    ),
                }
            )
    write_json(
        {"bar_cells": bar_rows, "paired_ctx_minus_alone": paired, "pipeline_check": pipeline},
        run / "metrics_decision_cell.json",
    )

    def strat(detector: str, ood: str) -> float | None:
        return next(
            r["auroc_stratified"]
            for r in bar_rows
            if (r["detector"], r["ood"], r["ref"]) == (detector, ood, "wildchat_en_test")
        )

    length_only = {o: strat("length", o) for o in TEST_POOLS if o != "wildchat_en_test"}
    measurable = {o: v is not None and 0.45 <= v <= 0.55 for o, v in length_only.items()}
    verdicts = {}
    for d in BAR_DETECTORS:
        v = {o: strat(d, o) for o in TEST_POOLS if o != "wildchat_en_test"}
        b = {o: band(v[o]) if measurable[o] else "unmeasurable" for o in v}
        control = {
            "high": "format_artifact",
            "low": "mix_multiturn_read_as_covered",
            "mid": "ambiguous",
            "unmeasurable": "unmeasurable",
        }[b["dolci_mt_test"]]
        if b["dolci_mt_test"] != "low":
            q1 = "not_readable_control_failed"
        elif (b["wc_mt_ctx"], b["wc_mt_alone"]) == ("high", "low"):
            q1 = "context_makes_later_turn_a_gap"
        elif (b["wc_mt_ctx"], b["wc_mt_alone"]) == ("high", "high"):
            q1 = "later_turn_messages_are_a_gap"
        elif (b["wc_mt_ctx"], b["wc_mt_alone"]) == ("low", "low"):
            q1 = "not_a_gap"
        else:
            q1 = "ambiguous" if "unmeasurable" not in (b["wc_mt_ctx"], b["wc_mt_alone"]) else "unmeasurable"
        q2 = {
            o: {"high": "flagged_as_gap", "low": "missed", "mid": "ambiguous", "unmeasurable": "unmeasurable"}[b[o]]
            for o in ("tool_ctx", "tool_alone")
        }
        verdicts[d] = {
            "stratified_vs_wildchat": v,
            "control": control,
            "q1_real_multiturn": q1,
            "q2_tool_use": q2,
            "tfidf_beside": {o: strat("tfidf", o) for o in v},
        }
    decision = {
        "cell": "ref idfit_mt (exp01 idfit + dolci_mt_fit), cmean L16, dolma full, vs wildchat_en_test, exp01 coarse bands",
        "pipeline_check": {
            "spearman_with_exp01_wildchat_scores": pipeline,
            "threshold": ">= 0.99",
            "cleared": is_pipeline_ok,
        },
        "measurability": {"length_only_stratified": length_only, "measurable": measurable},
        "verdicts": verdicts if is_pipeline_ok else None,
        "paired_ctx_minus_alone": paired,
    }
    write_json(decision, run / "decision.json")
    write_jsonl(
        [
            {
                "id": str(feats[pool]["ids"][i]),
                "pool": pool,
                "n_content_tokens": int(lengths[pool][i]),
                **{f"{d}_idfit_mt": float(main_scores[pool][d][i]) for d in main_scores[pool]},
                **{f"{d}_idfit": float(cells[("idfit", *CELL)][pool][d][i]) for d in BAR_DETECTORS},
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
