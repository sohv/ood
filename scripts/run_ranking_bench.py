# exp11: tests length removal, pooling, per-cluster aggregation and known-ood rules for gap-discovery rankings on planted needles.
# uv run --no-sync -m scripts.run_ranking_bench --run_dir results/raw/261019_exp11_ranking_bench_v1 --wildchat_features results/raw/261018_exp06_density_ratio_v1/features/deploy.npz --wildchat_pool data/processed/exp06/deploy.jsonl --lmsys_features results/raw/261018_exp09_discovery_v1/features/lmsys.npz --lmsys_pool data/processed/exp09/lmsys.jsonl --needles_dir data/processed/exp06 --exp01_dir results/raw/261018_exp01_repro_v1 --exp01_pools data/processed/exp01 --seed 0

import os

os.environ.setdefault("OMP_NUM_THREADS", "16")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "16")

import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import simple_parsing
from scipy.stats import spearmanr

from src.data.io import load_jsonl
from src.interp.features import LAYERS
from src.metrics.density_ratio import length_controlled
from src.metrics.detectors import KNN, Mahalanobis, tfidf_mix_vs_dolma
from src.metrics.io import write_json, write_jsonl
from src.metrics.ranking import (
    cluster_order,
    kmeans_labels,
    known_gate,
    length_bands,
    nearest_in_log_length,
    spline_residual,
)
from src.utils.git import get_git_hash
from src.utils.logging import setup_logging
from src.utils.seed import set_seed

LOGGER = logging.getLogger(__name__)

TYPES = ("legal", "medical", "aya", "magicoder")
GAP, CONTROL = ("legal", "medical"), ("aya", "magicoder")
Q2_LAYERS = (8, 16, 24, 32)
POOLINGS = ("cmean", "cmax", "clast")
CELL = ("cmean", 16)
Q1_ARMS = ("dknn", "dmaha", "knn", "maha", "tfidf", "length", "random")
BAND_ARMS = ("dknn", "dmaha", "knn", "maha")
LOOP_BASES = ("dknn", "dmaha")
LOOP_RULES = ("none", "add_to_id", "known_gate")
SURFACE_MIN = 5
NEVER = 99


@dataclass
class BenchConfig:
    run_dir: str = ""
    wildchat_features: str = ""
    wildchat_pool: str = ""
    lmsys_features: str = ""
    lmsys_pool: str = ""
    needles_dir: str = ""
    needles_features: str = ""
    exp01_dir: str = ""
    exp01_pools: str = ""
    seed: int = 0
    needle_seeds: list[int] = field(default_factory=lambda: [0, 1, 2])
    n_per_type: int = 100
    top_k: int = 100
    loop_rounds: int = 5
    loop_k: int = 200
    kmeans_k: int = 200
    kmeans_k_alt: int = 50
    kmeans_n_init: int = 3
    n_bands: int = 5
    band_ref: int = 500
    shrinkage: float = 0.5
    knn_k: int = 10
    dolma_tfidf_words: int = 400


def load_pool(feat_path: Path, pool_path: Path, cells: list[tuple[str, int]]) -> dict:
    f = np.load(feat_path)
    rows = load_jsonl(pool_path)
    assert list(f["ids"]) == [r["id"] for r in rows], f"{feat_path} rows misaligned with {pool_path}"
    out = {c: f[c[0]][:, LAYERS.index(c[1])].astype(np.float32) for c in cells}
    out["len"] = f["n_content_tokens"].astype(int)
    out["text"] = [r["text"] for r in rows]
    out["language"] = [r.get("language", "") for r in rows]
    return out


def concat(parts: list[dict]) -> dict:
    return {
        k: (
            np.concatenate([p[k] for p in parts])
            if isinstance(parts[0][k], np.ndarray)
            else [item for p in parts for item in p[k]]
        )
        for k in parts[0]
    }


def take(pool: dict, idx: np.ndarray) -> dict:
    return {k: (v[idx] if isinstance(v, np.ndarray) else [v[i] for i in idx]) for k, v in pool.items()}


def jitter_order(scores: np.ndarray, seed: int) -> np.ndarray:
    return np.argsort(-(scores + np.random.default_rng(seed).random(len(scores)) * 1e-9), kind="stable")


def top_metrics(order: np.ndarray, types: np.ndarray, lengths: np.ndarray, k: int) -> dict:
    top = order[:k]
    out = {"top_median_len_ratio": float(np.median(lengths[top]) / np.median(lengths))}
    for name, members in (("gap", GAP), ("control", CONTROL), *((t, (t,)) for t in TYPES)):
        is_t = np.isin(types, members)
        out[f"{name}_p@{k}"] = float(is_t[top].mean())
        out[f"{name}_uplift@{k}"] = float(is_t[top].mean() / is_t.mean())
    return out


def compare(value: float, base: float, len_ratios: list[float], control: float) -> str:
    is_balanced = all(0.67 <= r <= 1.5 for r in len_ratios)
    if value >= 1.5 * base and value >= base + 2 and is_balanced and control <= 2:
        return "better"
    if value <= base / 1.5:
        return "worse"
    return "equivalent"


def main():
    config = simple_parsing.parse(BenchConfig)
    missing = [k for k, v in asdict(config).items() if v == "" and k != "needles_features"]
    assert not missing, f"required arguments not set: {missing}"
    run = Path(config.run_dir)
    setup_logging(run)
    set_seed(config.seed)
    # run_extract already wrote config.json when the needle features were extracted into this directory
    config_name = "bench_config.json" if (run / "config.json").exists() else "config.json"
    write_json({"git_hash": get_git_hash(), **asdict(config)}, run / config_name)
    needle_feats = Path(config.needles_features) if config.needles_features else run / "features"
    cells = sorted({(p, l) for p in POOLINGS for l in Q2_LAYERS} | {CELL})

    e1f, e1p = Path(config.exp01_dir) / "features", Path(config.exp01_pools)
    deploy = concat(
        [
            load_pool(Path(config.wildchat_features), Path(config.wildchat_pool), cells),
            load_pool(Path(config.lmsys_features), Path(config.lmsys_pool), cells),
        ]
    )
    n_wc = len(load_jsonl(config.wildchat_pool))
    deploy["source"] = np.array(["wildchat"] * n_wc + ["lmsys"] * (len(deploy["len"]) - n_wc))
    deploy["type"] = np.array(["deploy"] * len(deploy["len"]))
    banks = {}
    for t in TYPES:
        banks[t] = load_pool(needle_feats / f"needles_{t}.npz", Path(config.needles_dir) / f"needles_{t}.jsonl", cells)
        banks[t]["source"] = np.array(["needle"] * len(banks[t]["len"]))
        banks[t]["type"] = np.array([t] * len(banks[t]["len"]))
    idfit = load_pool(e1f / "idfit.npz", e1p / "idfit.jsonl", cells)
    mix = concat([idfit, load_pool(e1f / "idmix.npz", e1p / "idmix.jsonl", cells)])
    dolma = load_pool(e1f / "dolma_full.npz", e1p / "dolma.jsonl", cells)
    LOGGER.info(f"deploy {len(deploy['len'])} rows, mix {len(mix['len'])}, dolma {len(dolma['len'])}")

    # fixed scores on deploy and on every bank, so planted pools are assembled by indexing
    def score_sets(cell: tuple[str, int], with_maha: bool) -> dict[str, dict[str, np.ndarray]]:
        scale = idfit[cell].std(0) + 1e-6
        knn_mix, knn_dolma = KNN(mix[cell], scale, config.knn_k), KNN(dolma[cell], scale, config.knn_k)
        if with_maha:
            maha_mix = Mahalanobis(mix[cell], shrinkage=config.shrinkage)
            maha_dolma = Mahalanobis(dolma[cell], shrinkage=config.shrinkage)
        out = {}
        for name, src in (("deploy", deploy), *banks.items()):
            x = src[cell]
            k = knn_mix(x)
            out[name] = {"knn": k, "dknn": k - knn_dolma(x)}
            if with_maha:
                m = maha_mix(x)
                out[name] |= {"maha": m, "dmaha": m - maha_dolma(x)}
        return out

    fixed = {cell: score_sets(cell, with_maha=(cell == CELL)) for cell in cells}
    LOGGER.info("fixed dense scores done")
    tfidf = tfidf_mix_vs_dolma(
        idfit["text"], [" ".join(t.split()[: config.dolma_tfidf_words]) for t in dolma["text"]], config.seed
    )
    for name, src in (("deploy", deploy), *banks.items()):
        fixed[CELL][name]["tfidf"] = tfidf(src["text"])
        fixed[CELL][name]["length"] = src["len"].astype(float)
    LOGGER.info("tfidf done")

    def planted(s: int) -> dict:
        parts = [("deploy", np.arange(len(deploy["len"])))] + [
            (t, np.arange(s * config.n_per_type, (s + 1) * config.n_per_type)) for t in TYPES
        ]
        srcs = {"deploy": deploy, **banks}
        pool = concat([take(srcs[n], idx) for n, idx in parts])
        for cell in cells:
            for arm in fixed[cell]["deploy"]:
                pool[(cell, arm)] = np.concatenate([fixed[cell][n][arm][idx] for n, idx in parts])
        pool[(CELL, "random")] = np.random.default_rng(1000 + s).random(len(pool["len"]))
        assert len(pool["len"]) == len(deploy["len"]) + len(TYPES) * config.n_per_type
        return pool

    def bandfit_scores(pool: dict) -> dict[str, np.ndarray]:
        bands = length_bands(pool["len"], config.n_bands)
        out = {arm: np.zeros(len(pool["len"])) for arm in BAND_ARMS}
        for b in np.unique(bands):
            rows = np.where(bands == b)[0]
            target = float(np.median(np.log(pool["len"][rows].astype(float))))
            m = mix[CELL][nearest_in_log_length(mix["len"], target, config.band_ref)]
            d = dolma[CELL][nearest_in_log_length(dolma["len"], target, config.band_ref)]
            scale = m.std(0) + 1e-6
            x = pool[CELL][rows]
            k, kd = KNN(m, scale, config.knn_k)(x), KNN(d, scale, config.knn_k)(x)
            mh, md = Mahalanobis(m, shrinkage=config.shrinkage)(x), Mahalanobis(d, shrinkage=config.shrinkage)(x)
            out["knn"][rows], out["dknn"][rows] = k, k - kd
            out["maha"][rows], out["dmaha"][rows] = mh, mh - md
        return out

    def ranked(scores: np.ndarray, lengths: np.ndarray, ranking: str) -> np.ndarray:
        if ranking == "pct20":
            return length_controlled(scores, lengths)
        if ranking == "spline":
            return spline_residual(scores, lengths)
        return scores

    def row(pool: dict, scores: np.ndarray, s: int, **meta) -> dict:
        order = jitter_order(scores, s)
        return {
            **meta,
            "needle_seed": s,
            "spearman_log_len": float(spearmanr(scores, np.log(pool["len"])).statistic),
            **top_metrics(order, pool["type"], pool["len"], config.top_k),
        }

    q1, q2, q3, loop = [], [], [], []
    for s in config.needle_seeds:
        pool = planted(s)
        band = bandfit_scores(pool)
        for arm in Q1_ARMS:
            for ranking in ("raw", "pct20", "spline"):
                q1.append(row(pool, ranked(pool[(CELL, arm)], pool["len"], ranking), s, arm=arm, ranking=ranking))
        for arm in BAND_ARMS:
            q1.append(row(pool, band[arm], s, arm=arm, ranking="bandfit"))
            q1.append(row(pool, length_controlled(band[arm], pool["len"]), s, arm=arm, ranking="bandfit_pct20"))
        for p in POOLINGS:
            for l in Q2_LAYERS:
                q2.append(
                    row(
                        pool,
                        length_controlled(pool[((p, l), "dknn")], pool["len"]),
                        s,
                        pooling=p,
                        layer=l,
                        arm="dknn",
                        ranking="pct20",
                    )
                )
        base = length_controlled(pool[(CELL, "dknn")], pool["len"])
        for k in (config.kmeans_k, config.kmeans_k_alt):
            labels = kmeans_labels(pool[CELL], k, config.seed, config.kmeans_n_init)
            order = cluster_order(base, labels, config.top_k)
            assert len(set(order.tolist())) == config.top_k
            q3.append(
                {
                    "needle_seed": s,
                    "kmeans_k": k,
                    "arm": "dknn",
                    "ranking": "pct20_cluster",
                    **top_metrics(order, pool["type"], pool["len"], config.top_k),
                }
            )
        q3.append(
            {
                "needle_seed": s,
                "kmeans_k": None,
                "arm": "dknn",
                "ranking": "pct20_row",
                **top_metrics(jitter_order(base, s), pool["type"], pool["len"], config.top_k),
            }
        )
        LOGGER.info(f"seed {s}: q1-q3 done; dknn pct20 gap uplift {q3[-1]['gap_uplift@100']:.2f}")

        if s == config.needle_seeds[0]:
            order = jitter_order(base, s)
            shown = [i for i in order if pool["type"][i] == "deploy"][:10]
            write_json(
                [
                    {
                        "id": f"top_{j:02d}",
                        "source": str(pool["source"][i]),
                        "language": pool["language"][i],
                        "n_tokens": int(pool["len"][i]),
                        "text": pool["text"][i][:300],
                    }
                    for j, i in enumerate(shown)
                ],
                run / "top10_nonneedle_seed0.json",
            )
            write_jsonl(
                [
                    {
                        "id": f"seed0_{i:05d}",
                        "type": str(pool["type"][i]),
                        "source": str(pool["source"][i]),
                        "n_tokens": int(pool["len"][i]),
                        **{arm: float(pool[(CELL, arm)][i]) for arm in Q1_ARMS},
                        "dknn_pct20": float(base[i]),
                        "dknn_bandfit": float(band["dknn"][i]),
                    }
                    for i in range(len(pool["len"]))
                ],
                run / "generations" / "scores_seed0.jsonl",
            )

        scale = idfit[CELL].std(0) + 1e-6
        x = pool[CELL]
        knn_dolma = pool[(CELL, "knn")] - pool[(CELL, "dknn")]
        maha_dolma = pool[(CELL, "maha")] - pool[(CELL, "dmaha")]
        for base_name in LOOP_BASES:
            for rule in LOOP_RULES:
                remaining = np.ones(len(x), dtype=bool)
                confirmed: list[int] = []
                found = {t: 0 for t in TYPES}
                dominant = None
                for rnd in range(1, config.loop_rounds + 1):
                    idx = np.where(remaining)[0]
                    if rule == "add_to_id" and confirmed:
                        ref = np.vstack([mix[CELL], x[confirmed]])
                        sc = (
                            (KNN(ref, scale, config.knn_k)(x[idx]) - knn_dolma[idx])
                            if base_name == "dknn"
                            else (Mahalanobis(ref, shrinkage=config.shrinkage)(x[idx]) - maha_dolma[idx])
                        )
                    else:
                        sc = pool[(CELL, base_name)][idx]
                    lc = length_controlled(sc, pool["len"][idx])
                    tau, n_gated = None, 0
                    if rule == "known_gate" and len(confirmed) >= SURFACE_MIN:
                        drop, tau = known_gate(x[idx], x[confirmed], scale, config.knn_k)
                        lc = np.where(drop, -np.inf, lc)
                        n_gated = int(drop.sum())
                    top = idx[jitter_order(lc, rnd)[: config.loop_k]]
                    remaining[top] = False
                    top_types = pool["type"][top]
                    confirmed += top[top_types != "deploy"].tolist()
                    for t in TYPES:
                        found[t] += int((top_types == t).sum())
                    if dominant is None and rnd == 1:
                        counts = {t: found[t] for t in TYPES}
                        dominant = max(counts, key=counts.get) if max(counts.values()) > 0 else None
                    loop.append(
                        {
                            "base": base_name,
                            "rule": rule,
                            "needle_seed": s,
                            "round": rnd,
                            "reviewed": int((~remaining).sum()),
                            "n_confirmed": len(confirmed),
                            "tau": tau,
                            "n_gated": n_gated,
                            **{f"{t}_recall": found[t] / config.n_per_type for t in TYPES},
                            "gap_recall": (found["legal"] + found["medical"]) / (2 * config.n_per_type),
                            "dominant_type": dominant,
                            "dominant_share": float((top_types == dominant).mean()) if dominant else None,
                            "top_lmsys_share": float(np.mean(pool["source"][top] == "lmsys")),
                            "top_median_len_ratio": float(np.median(pool["len"][top]) / np.median(pool["len"])),
                        }
                    )
                assert (~remaining).sum() == config.loop_rounds * config.loop_k
                LOGGER.info(f"loop seed {s} {base_name} {rule}: gap recall {loop[-1]['gap_recall']:.3f}, found {found}")
    write_json(q1, run / "metrics_q1_length.json")
    write_json(q2, run / "metrics_q2_pooling.json")
    write_json(q3, run / "metrics_q3_cluster.json")
    write_json(loop, run / "loop.json")

    def mean_of(rows: list[dict], key: str, **where) -> float:
        vals = [r[key] for r in rows if all(r.get(k) == v for k, v in where.items())]
        assert len(vals) == len(config.needle_seeds), (where, len(vals))
        return float(np.mean(vals))

    def list_of(rows: list[dict], key: str, **where) -> list[float]:
        return [r[key] for r in rows if all(r.get(k) == v for k, v in where.items())]

    def verdict_q1(ranking: str) -> dict:
        w = {"arm": "dknn", "ranking": ranking}
        base = mean_of(q1, "gap_uplift@100", arm="dknn", ranking="pct20")
        value, control = mean_of(q1, "gap_uplift@100", **w), mean_of(q1, "control_uplift@100", **w)
        ratios = list_of(q1, "top_median_len_ratio", **w)
        return {
            "gap_uplift": value,
            "pct20_gap_uplift": base,
            "control_uplift": control,
            "len_ratios": ratios,
            "verdict": compare(value, base, ratios, control),
        }

    def verdict_q2(pooling: str) -> dict:
        w = {"pooling": pooling, "layer": 16}
        base = mean_of(q2, "gap_uplift@100", pooling="cmean", layer=16)
        value, control = mean_of(q2, "gap_uplift@100", **w), mean_of(q2, "control_uplift@100", **w)
        ratios = list_of(q2, "top_median_len_ratio", **w)
        return {
            "gap_uplift": value,
            "cmean_gap_uplift": base,
            "control_uplift": control,
            "len_ratios": ratios,
            "verdict": compare(value, base, ratios, control),
        }

    def verdict_q3() -> dict:
        base = mean_of(q3, "gap_uplift@100", ranking="pct20_row")
        value = mean_of(q3, "gap_uplift@100", kmeans_k=config.kmeans_k)
        control = mean_of(q3, "control_uplift@100", kmeans_k=config.kmeans_k)
        ratios = list_of(q3, "top_median_len_ratio", kmeans_k=config.kmeans_k)
        return {
            "gap_uplift": value,
            "row_gap_uplift": base,
            "control_uplift": control,
            "len_ratios": ratios,
            "verdict": compare(value, base, ratios, control),
        }

    def surface_rounds(base_name: str, rule: str, s: int) -> dict[str, int]:
        rows = sorted(
            [r for r in loop if (r["base"], r["rule"], r["needle_seed"]) == (base_name, rule, s)],
            key=lambda r: r["round"],
        )
        return {
            t: next((r["round"] for r in rows if r[f"{t}_recall"] * config.n_per_type >= SURFACE_MIN), NEVER)
            for t in GAP
        }

    def final_recall(base_name: str, rule: str, s: int) -> float:
        return next(
            r["gap_recall"]
            for r in loop
            if (r["base"], r["rule"], r["needle_seed"], r["round"]) == (base_name, rule, s, config.loop_rounds)
        )

    def verdict_q4(base_name: str) -> dict:
        per_seed = []
        for s in config.needle_seeds:
            second = {rule: max(surface_rounds(base_name, rule, s).values()) for rule in LOOP_RULES}
            recall = {rule: final_recall(base_name, rule, s) for rule in LOOP_RULES}
            per_seed.append(
                {
                    "needle_seed": s,
                    "second_gap_surface_round": second,
                    "gap_recall_round5": recall,
                    "surface_rounds": {rule: surface_rounds(base_name, rule, s) for rule in LOOP_RULES},
                }
            )
        earlier = sum(
            p["second_gap_surface_round"]["known_gate"] < p["second_gap_surface_round"]["add_to_id"] for p in per_seed
        )
        identical = all(
            p["second_gap_surface_round"]["known_gate"] == p["second_gap_surface_round"]["add_to_id"] for p in per_seed
        )
        recall_ok = all(
            p["gap_recall_round5"]["known_gate"] >= p["gap_recall_round5"]["add_to_id"] - 0.05 for p in per_seed
        )
        if earlier >= 2 and recall_ok:
            verdict = "known_gate_better"
        elif identical:
            verdict = "equivalent"
        else:
            verdict = "ambiguous"
        return {
            "per_seed": per_seed,
            "known_gate_earlier_on_seeds": earlier,
            "recall_within_0.05": recall_ok,
            "verdict": verdict,
        }

    random_uplifts = list_of(q1, "gap_uplift@100", arm="random", ranking="raw")
    decision = {
        "cell": "cmean L16, dolma full, mix = exp01 idfit + idmix, needles 0.5% per type in the 20,000-row wildchat + lmsys pool",
        "q1_length_removal": {r: verdict_q1(r) for r in ("raw", "spline", "bandfit")},
        "q2_pooling": {p: verdict_q2(p) for p in ("clast", "cmax")},
        "q3_cluster_aggregation": verdict_q3(),
        "q4_known_ood_rule": {b: verdict_q4(b) for b in LOOP_BASES},
        "sanity": {
            "random_gap_uplift_per_seed": random_uplifts,
            "random_mean_in_band": 0 <= float(np.mean(random_uplifts)) <= 3,
            "planted_rows_per_seed": len(deploy["len"]) + len(TYPES) * config.n_per_type,
            "loop_reviewed_per_group": config.loop_rounds * config.loop_k,
        },
    }
    write_json(decision, run / "decision.json")
    print(f"decision written to {run / 'decision.json'}")


if __name__ == "__main__":
    main()
