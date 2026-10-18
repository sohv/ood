# exp04: re-scores reddit pools (seeds 0-2) with fine length bands and tests detector scores against measured coverage.
# uv run -m scripts.run_exp04_analysis --output_dir results/raw/261018_exp04_rerun_v1 --exp01_dir results/raw/261018_exp01_repro_v1 --exp01_pools data/processed/exp01 --coverage_path results/raw/261018_exp04_coverage_v1/coverage.jsonl --seed 0

import os

os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "8")

import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import simple_parsing
from scipy.stats import spearmanr

from src.data.io import load_jsonl
from src.interp.features import LAYERS
from src.metrics.detectors import LENGTH_BANDS, KNN, Mahalanobis, auroc, bootstrap_ci, partial_spearman, stratified_auroc, tfidf_mix_vs_dolma
from src.metrics.io import write_json
from src.utils.git import get_git_hash
from src.utils.logging import setup_logging
from src.utils.seed import set_seed

LOGGER = logging.getLogger(__name__)

COVERED = ["reddit_relationship_advice", "reddit_fitness", "reddit_cooking", "reddit_writing", "reddit_relationships", "reddit_learnprogramming", "reddit_askscience"]
REDDIT = [*COVERED, "legal_rtb"]
DETECTORS = ("maha", "knn", "dmaha_full", "dknn_full", "tfidf", "length")
BAR_DETECTORS = ("dmaha_full", "dknn_full")
FINE_BANDS = tuple(float(b) for b in np.unique(np.round(2 ** (np.arange(0, 41) / 4)))) + (1025.0,)


@dataclass
class Exp04Config:
    output_dir: str = ""
    exp01_dir: str = ""
    exp01_pools: str = ""
    coverage_path: str = ""
    seed: int = 0
    n_boot: int = 1000
    seed_runs: list[str] = field(default_factory=lambda: [
        "0:results/raw/261018_exp03_reddit_v1:data/processed/exp03",
        "1:results/raw/261018_exp04_seed1_v1:data/processed/exp04_seed1",
        "2:results/raw/261018_exp04_seed2_v1:data/processed/exp04_seed2",
    ])


def main():
    config = simple_parsing.parse(Exp04Config)
    out = Path(config.output_dir)
    setup_logging(out)
    set_seed(config.seed)
    write_json({"git_hash": get_git_hash(), **asdict(config), "fine_bands": FINE_BANDS}, out / "config.json")
    li = LAYERS.index(16)
    ref = {n: np.load(Path(config.exp01_dir) / "features" / f"{n}.npz") for n in ["idfit", "dolma_full", "idmix", "wildchat_en_test", "legal", "medical"]}
    x = {n: f["cmean"][:, li].astype(np.float32) for n, f in ref.items()}
    scale = x["idfit"].std(0) + 1e-6
    maha_mix, knn_mix = Mahalanobis(x["idfit"]), KNN(x["idfit"], scale)
    maha_d, knn_d = Mahalanobis(x["dolma_full"]), KNN(x["dolma_full"], scale)
    texts = {n: [r["text"] for r in load_jsonl(Path(config.exp01_pools) / f"{n}.jsonl")] for n in ["idfit", "dolma", "idmix", "wildchat_en_test", "legal", "medical"]}
    tfidf = tfidf_mix_vs_dolma(texts["idfit"], [" ".join(t.split()[:400]) for t in texts["dolma"]], config.seed)
    coverage = {r["id"] + "|" + r["source"].split("/")[0]: r for r in load_jsonl(config.coverage_path)}

    def score(feats: np.ndarray, txt: list[str], length: np.ndarray) -> dict[str, np.ndarray]:
        m, k = maha_mix(feats), knn_mix(feats)
        return {"maha": m, "knn": k, "dmaha_full": m - maha_d(feats), "dknn_full": k - knn_d(feats), "tfidf": tfidf(txt), "length": length.astype(float)}

    q1, q2, pool_level, balance = {}, {}, {}, {}
    for spec in config.seed_runs:
        seed, run_dir, pool_dir = spec.split(":")
        feats = {n: np.load(Path(run_dir) / "features" / f"{n}.npz") for n in REDDIT}
        recs = {n: load_jsonl(Path(pool_dir) / f"{n}.jsonl") for n in REDDIT}
        for n in REDDIT:
            assert list(feats[n]["ids"]) == [r["id"] for r in recs[n]], f"seed {seed} {n} misaligned"
        lengths = {n: feats[n]["n_content_tokens"] for n in REDDIT}
        scores = {n: score(feats[n]["cmean"][:, li].astype(np.float32), [r["text"] for r in recs[n]], lengths[n]) for n in REDDIT}
        cov = {n: np.array([coverage[f"{r['id']}|{Path(pool_dir).name}"]["cov_minilm"] for r in recs[n]]) for n in REDDIT}
        cov_t = {n: np.array([coverage[f"{r['id']}|{Path(pool_dir).name}"]["cov_tfidf"] for r in recs[n]]) for n in REDDIT}

        rows = []
        for d in DETECTORS:
            for sub in COVERED:
                o, r, lo, lr = scores["legal_rtb"][d], scores[sub][d], lengths["legal_rtb"], lengths[sub]
                strat = stratified_auroc(o, r, lo, lr, FINE_BANDS)
                row = {"detector": d, "ref": sub, "auroc_stratified_fine": strat["auroc"], "n_bands": strat["n_bands"], "coverage": strat["coverage"],
                       "auroc_stratified_coarse": stratified_auroc(o, r, lo, lr, LENGTH_BANDS)["auroc"], "auroc_raw": auroc(o, r)}
                if d in BAR_DETECTORS:
                    row["ci_fine"] = bootstrap_ci(lambda i, j: stratified_auroc(o[i], r[j], lo[i], lr[j], FINE_BANDS)["auroc"], len(o), len(r), config.n_boot, config.seed)
                rows.append(row)
        balance[seed] = {sub: next(r["auroc_stratified_fine"] for r in rows if (r["detector"], r["ref"]) == ("length", sub)) for sub in COVERED}
        medians = {d: float(np.median([r["auroc_stratified_fine"] for r in rows if r["detector"] == d and r["auroc_stratified_fine"] is not None])) for d in DETECTORS}
        q1[seed] = {"rows": rows, "median_fine": medians}

        all_len = np.concatenate([lengths[n] for n in REDDIT])
        all_cov, all_cov_t = np.concatenate([cov[n] for n in REDDIT]), np.concatenate([cov_t[n] for n in REDDIT])
        q2[seed] = {d: {"partial_spearman_minilm": partial_spearman(np.concatenate([scores[n][d] for n in REDDIT]), all_cov, all_len),
                        "partial_spearman_tfidf": partial_spearman(np.concatenate([scores[n][d] for n in REDDIT]), all_cov_t, all_len)}
                    for d in DETECTORS if d != "length"}
        q2[seed]["length"] = {"spearman_minilm": float(spearmanr(all_len, all_cov)[0])}
        pool_level[seed] = {"median_cov_minilm": {n: float(np.median(cov[n])) for n in REDDIT},
                            **{d: float(spearmanr([np.median(cov[n]) for n in REDDIT], [np.median(scores[n][d]) for n in REDDIT])[0]) for d in DETECTORS}}
        LOGGER.info(f"seed {seed} done: medians {medians}")

    def q2_verdict(d: str) -> str:
        rhos = [q2[s][d]["partial_spearman_minilm"] for s in q2]
        return "tracks_coverage" if all(r <= -0.30 for r in rhos) else ("does_not_track" if all(abs(r) < 0.10 for r in rhos) else "ambiguous")

    def q1_verdict(m: float) -> str:
        return "separates_coverage" if m >= 0.75 else ("reads_register" if m <= 0.60 else "ambiguous")

    decision = {
        "q1_exp03_question_fine_bands": {s: {d: {"median": q1[s]["median_fine"][d], "verdict": q1_verdict(q1[s]["median_fine"][d])} for d in BAR_DETECTORS} for s in q1},
        "length_balance_check": {s: {"values": v, "all_within_0.45_0.55": all(0.45 <= x <= 0.55 for x in v.values() if x is not None)} for s, v in balance.items()},
        "q2_score_vs_measured_coverage": {d: {"partial_spearman_minilm_by_seed": {s: q2[s][d]["partial_spearman_minilm"] for s in q2}, "verdict": q2_verdict(d)} for d in BAR_DETECTORS},
    }

    exp01_fine = {}
    ref_scores = {n: score(x[n], texts[n], ref[n]["n_content_tokens"]) for n in ["wildchat_en_test", "legal", "medical"]}
    for d in DETECTORS:
        for pool in ("legal", "medical"):
            o, r = ref_scores[pool][d], ref_scores["wildchat_en_test"][d]
            lo, lr = ref[pool]["n_content_tokens"], ref["wildchat_en_test"]["n_content_tokens"]
            fine = stratified_auroc(o, r, lo, lr, FINE_BANDS)
            exp01_fine[f"{d}|{pool}_vs_wildchat"] = {"fine": fine["auroc"], "n_bands": fine["n_bands"], "coverage": fine["coverage"],
                                                     "coarse": stratified_auroc(o, r, lo, lr, LENGTH_BANDS)["auroc"]}

    write_json(decision, out / "decision.json")
    write_json(q1, out / "q1_fine_bands.json")
    write_json({"row_level": q2, "pool_level": pool_level}, out / "q2_coverage.json")
    write_json(exp01_fine, out / "exp01_fine_bands_exploratory.json")
    print(f"decision written to {out / 'decision.json'}")


if __name__ == "__main__":
    main()
