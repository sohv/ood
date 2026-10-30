# density_ratio_needles: plants gap and control needles in wildchat and tests whether a density-ratio score surfaces the gaps.
# uv run --no-sync -m scripts.run_density_ratio --run_dir results/raw/261018_density_ratio_needles_density_ratio_v1 --pools_dir data/processed/density_ratio_needles --repro_two_sided_dir results/raw/261018_repro_two_sided_repro_v1 --repro_two_sided_pools data/processed/repro_two_sided --seed 0

import os

os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "8")

import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import simple_parsing

from src.data.io import load_jsonl
from src.interp.features import LAYERS
from src.metrics.density_ratio import cross_fit_dense, cross_fit_tfidf, length_controlled, precision_at_k
from src.metrics.detectors import KNN, Mahalanobis
from src.metrics.io import write_json
from src.utils.git import get_git_hash
from src.utils.logging import setup_logging
from src.utils.seed import set_seed

LOGGER = logging.getLogger(__name__)

TYPES = ("legal", "medical", "aya", "magicoder")
GROUPS = {"gap": ("legal", "medical"), "control": ("aya", "magicoder")}
KS = (25, 50, 100, 200)


@dataclass
class DensityRatioConfig:
    run_dir: str = ""
    pools_dir: str = ""
    needles_pools_dir: str = ""
    needles_features_dir: str = ""
    repro_two_sided_dir: str = ""
    repro_two_sided_pools: str = ""
    seed: int = 0
    prevalences: list[float] = field(default_factory=lambda: [0.001, 0.003, 0.01])
    needle_seeds: list[int] = field(default_factory=lambda: [0, 1, 2])
    folds: int = 5
    c_dense: float = 0.1
    c_tfidf: float = 1.0
    loop_rounds: int = 5
    loop_k: int = 100


def main():
    config = simple_parsing.parse(DensityRatioConfig)
    run = Path(config.run_dir)
    setup_logging(run)
    set_seed(config.seed)
    write_json({"git_hash": get_git_hash(), **asdict(config)}, run / "detector_config.json")
    li = LAYERS.index(16)

    def load(feat_dir: Path, name: str, pool_dir: Path) -> dict:
        f = np.load(feat_dir / f"{name}.npz")
        recs = load_jsonl(pool_dir / f"{name}.jsonl")
        assert list(f["ids"]) == [r["id"] for r in recs], f"{name} rows misaligned"
        return {
            "cmean": f["cmean"][:, li].astype(np.float32),
            "clast": f["clast"][:, li].astype(np.float32),
            "len": f["n_content_tokens"],
            "text": [r["text"] for r in recs],
        }

    e1f, e1p = Path(config.repro_two_sided_dir) / "features", Path(config.repro_two_sided_pools)
    mix_parts = [load(e1f, n, e1p) for n in ("idfit", "idmix")]
    mix = {
        k: np.concatenate([p[k] for p in mix_parts]) if k != "text" else sum((p[k] for p in mix_parts), [])
        for k in ("cmean", "clast", "len", "text")
    }
    dolma = np.load(e1f / "dolma_full.npz")["cmean"][:, li].astype(np.float32)
    deploy = load(run / "features", "deploy", Path(config.pools_dir))
    needle_feats = Path(config.needles_features_dir) if config.needles_features_dir else run / "features"
    needle_pools = Path(config.needles_pools_dir) if config.needles_pools_dir else Path(config.pools_dir)
    banks = {t: load(needle_feats, f"needles_{t}", needle_pools) for t in TYPES}

    idfit = mix_parts[0]["cmean"]
    scale = idfit.std(0) + 1e-6
    maha = Mahalanobis(mix["cmean"], shrinkage=0.5)
    knn_mix, knn_dolma = KNN(mix["cmean"], scale), KNN(dolma, scale)

    def fixed_scores(x: np.ndarray) -> dict[str, np.ndarray]:
        k = knn_mix(x)
        return {"maha": maha(x), "knn": k, "dknn_full": k - knn_dolma(x)}

    fixed = {"deploy": fixed_scores(deploy["cmean"]), **{t: fixed_scores(banks[t]["cmean"]) for t in TYPES}}

    def planted(p: float, s: int) -> dict:
        n = int(round(p * len(deploy["len"])))
        parts = [("deploy", np.arange(len(deploy["len"])))] + [(t, np.arange(s * 100, s * 100 + n)) for t in TYPES]
        src = {"deploy": deploy, **banks}
        pool = {k: np.concatenate([src[name][k][idx] for name, idx in parts]) for k in ("cmean", "clast", "len")}
        pool["text"] = [src[name]["text"][i] for name, idx in parts for i in idx]
        pool["type"] = np.array([name for name, idx in parts for _ in idx])
        for arm in ("maha", "knn", "dknn_full"):
            pool[arm] = np.concatenate([fixed[name][arm][idx] for name, idx in parts])
        return pool

    def arm_scores(
        pool: dict, s: int, mix_cmean: np.ndarray, mix_clast: np.ndarray, mix_text: list[str], arms: tuple[str, ...]
    ) -> dict[str, np.ndarray]:
        out = {}
        if "dr_cmean" in arms:
            out["dr_cmean"] = cross_fit_dense(pool["cmean"], mix_cmean, config.c_dense, config.folds, s)
        if "dr_clast" in arms:
            out["dr_clast"] = cross_fit_dense(pool["clast"], mix_clast, config.c_dense, config.folds, s)
        if "dr_tfidf" in arms:
            out["dr_tfidf"] = cross_fit_tfidf(pool["text"], mix_text, config.c_tfidf, config.folds, s)
        return out

    rows = []
    for p in config.prevalences:
        for s in config.needle_seeds:
            pool = planted(p, s)
            scores = arm_scores(pool, s, mix["cmean"], mix["clast"], mix["text"], ("dr_cmean", "dr_clast", "dr_tfidf"))
            scores |= {a: pool[a] for a in ("maha", "knn", "dknn_full")}
            scores["length"] = pool["len"].astype(float)
            scores["random"] = np.random.default_rng(1000 + s).random(len(pool["len"]))
            if (p, s) == (0.01, config.needle_seeds[0]):
                lc_top = np.argsort(-length_controlled(scores["dr_cmean"], pool["len"]))[:15]
                write_json(
                    [
                        {
                            "id": f"top_{j:02d}",
                            "type": str(pool["type"][i]),
                            "n_tokens": int(pool["len"][i]),
                            "text": pool["text"][i][:300],
                        }
                        for j, i in enumerate(lc_top)
                    ],
                    run / "top15_dr_cmean.json",
                )
            for arm, sc in scores.items():
                for ranking, r in (("raw", sc), ("length_controlled", length_controlled(sc, pool["len"]))):
                    top100 = np.argsort(-(r + np.random.default_rng(s).random(len(r)) * 1e-9))[:100]
                    row = {
                        "arm": arm,
                        "prevalence": p,
                        "needle_seed": s,
                        "ranking": ranking,
                        "top100_median_len_ratio": float(np.median(pool["len"][top100]) / np.median(pool["len"])),
                    }
                    for g, types in {**GROUPS, **{t: (t,) for t in TYPES}}.items():
                        is_t = np.isin(pool["type"], types)
                        prev = is_t.mean()
                        for k in KS:
                            row[f"{g}_p@{k}"] = precision_at_k(r, is_t, k, s)
                        row[f"{g}_uplift@100"] = row[f"{g}_p@100"] / prev
                    rows.append(row)
            LOGGER.info(
                f"p={p} seed={s} done; dr_cmean lc gap uplift {next(r['gap_uplift@100'] for r in rows[::-1] if r['arm'] == 'dr_cmean' and r['ranking'] == 'length_controlled'):.1f}"
            )
    write_json(rows, run / "metrics.json")

    def mean_uplift(arm: str, p: float, group: str) -> float:
        return float(
            np.mean(
                [
                    r[f"{group}_uplift@100"]
                    for r in rows
                    if (r["arm"], r["prevalence"], r["ranking"]) == (arm, p, "length_controlled")
                ]
            )
        )

    judged = (0.003, 0.01)
    gap = {p: mean_uplift("dr_cmean", p, "gap") for p in judged}
    ctrl = {p: mean_uplift("dr_cmean", p, "control") for p in judged}
    ratio = {p: gap[p] / max(mean_uplift("dr_tfidf", p, "gap"), 1e-9) for p in judged}
    decision = {
        "q1_gap_uplift": {
            "values": gap,
            "verdict": "upsamples_gaps"
            if all(v >= 5 for v in gap.values())
            else ("does_not" if all(v <= 2 for v in gap.values()) else "ambiguous"),
        },
        "q2_control_uplift": {"values": ctrl, "verdict": "passes" if all(v <= 2 for v in ctrl.values()) else "fails"},
        "q3_dense_over_tfidf": {
            "values": ratio,
            "verdict": "activations_add"
            if all(v >= 1.5 for v in ratio.values())
            else ("no_gain" if all(v <= 1.0 for v in ratio.values()) else "ambiguous"),
        },
    }
    write_json(decision, run / "decision.json")

    loop = []
    for s in config.needle_seeds:
        for arm in ("dr_cmean", "dr_tfidf"):
            pool = planted(0.01, s)
            keep = np.ones(len(pool["len"]), dtype=bool)
            mix_c, mix_l, mix_t = mix["cmean"], mix["clast"], list(mix["text"])
            found = {t: 0 for t in TYPES}
            for rnd in range(config.loop_rounds):
                idx = np.where(keep)[0]
                sub = {k: (v[idx] if k != "text" else [v[i] for i in idx]) for k, v in pool.items()}
                sc = arm_scores(sub, s, mix_c, mix_l, mix_t, (arm,))[arm]
                lc = length_controlled(sc, sub["len"])
                top = idx[np.argsort(-(lc + np.random.default_rng(rnd).random(len(lc)) * 1e-9))[: config.loop_k]]
                for t in TYPES:
                    found[t] += int((pool["type"][top] == t).sum())
                keep[top] = False
                mix_c, mix_l = np.vstack([mix_c, pool["cmean"][top]]), np.vstack([mix_l, pool["clast"][top]])
                mix_t = mix_t + [pool["text"][i] for i in top]
                reviewed = (rnd + 1) * config.loop_k
                loop.append(
                    {
                        "arm": arm,
                        "needle_seed": s,
                        "round": rnd + 1,
                        "reviewed": reviewed,
                        "random_expected_recall": reviewed / len(pool["len"]),
                        **{f"{t}_recall": found[t] / 100 for t in TYPES},
                    }
                )
            LOGGER.info(f"loop seed={s} arm={arm}: {loop[-1]}")
    write_json(loop, run / "loop.json")
    print(f"decision written to {run / 'decision.json'}")


if __name__ == "__main__":
    main()
