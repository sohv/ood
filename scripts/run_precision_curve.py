# exp38: precision at K of the gap rankings (words, activations, relative frequency against the DPO and SFT+DPO mixes) on planted categories, as a curve over K.
# uv run -m scripts.run_precision_curve --run_dir results/raw/261021_exp38_precision_curve_v1 --wildchat_features results/raw/261018_exp06_density_ratio_v1/features/deploy.npz --wildchat_pool data/processed/exp06/deploy.jsonl --lmsys_features results/raw/261018_exp09_discovery_v1/features/lmsys.npz --lmsys_pool data/processed/exp09/lmsys.jsonl --needles_dir data/processed/exp06 --needles_features results/raw/261019_exp11_ranking_bench_v1/features --exp01_dir results/raw/261018_exp01_repro_v1 --exp01_pools data/processed/exp01 --traffic_emb results/raw/261021_exp31_self_nature_cells_v2/emb_traffic.npy --emb_v1 results/raw/261021_exp31_self_nature_cells_v1 --sft_path data/processed/exp31/dolci_sft_first_user.parquet --seed 0

import os

os.environ.setdefault("OMP_NUM_THREADS", "16")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "16")

import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import simple_parsing
import torch
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans

from scripts.run_ranking_bench import concat, jitter_order, load_pool, take
from src.data.io import load_jsonl
from src.data.pools import load_dolci_dpo, normalise
from src.metrics.density_ratio import length_controlled
from src.metrics.detectors import KNN, tfidf_mix_vs_dolma
from src.metrics.gap_pipeline import density_ratios_batched, text_hash
from src.metrics.io import write_json, write_jsonl
from src.metrics.ranking import cluster_order, kmeans_labels
from src.utils.git import get_git_hash
from src.utils.logging import setup_logging
from src.utils.seed import set_seed

LOGGER = logging.getLogger(__name__)

TYPES = ("legal", "medical", "aya", "magicoder")
GAP, CONTROL = ("legal", "medical"), ("aya", "magicoder")
CELLS = {"cmean": ("cmean", 16), "cmax": ("cmax", 16)}
EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
EMBED_REVISION = "e8f8c211226b894fcb81acc59f3b34ba3efd5f42"
ARMS = (
    "random",
    "word_pct20",
    "act_cmean_pct20",
    "act_cmax_pct20",
    "act_cmean_cluster",
    "rf_dpo_raw",
    "rf_dpo_pct20",
    "rf_dpo_cluster",
    "rf_full_raw",
    "rf_full_pct20",
    "rf_full_cluster",
)
RECALL_MARKS = (0.25, 0.5, 0.75)


@dataclass
class CurveConfig:
    run_dir: str = ""
    wildchat_features: str = ""
    wildchat_pool: str = ""
    lmsys_features: str = ""
    lmsys_pool: str = ""
    needles_dir: str = ""
    needles_features: str = ""
    exp01_dir: str = ""
    exp01_pools: str = ""
    traffic_emb: str = ""
    emb_v1: str = ""
    sft_path: str = ""
    seed: int = 0
    needle_seeds: list[int] = field(default_factory=lambda: [0, 1, 2])
    n_per_type: int = 100
    ks: list[int] = field(default_factory=lambda: [25, 50, 100, 200, 500, 1000, 2000, 5000])
    knn_k: int = 10
    rf_k: int = 50
    kmeans_k: int = 200
    kmeans_n_init: int = 3
    min_cluster: int = 10
    embed_chars: int = 1000
    dolma_tfidf_words: int = 400
    n_embed_check: int = 50


def planted_uplift(precision: float, base_rate: float) -> float:
    return float(precision / base_rate)


def curve_rows(order: np.ndarray, types: np.ndarray, lengths: np.ndarray, ks: list[int], **meta) -> list[dict]:
    is_gap, is_control = np.isin(types, GAP), np.isin(types, CONTROL)
    out = []
    for k in ks:
        top = order[:k]
        row = {
            **meta,
            "k": k,
            "gap_precision": float(is_gap[top].mean()),
            "gap_uplift": planted_uplift(is_gap[top].mean(), is_gap.mean()),
            "gap_recall": float(is_gap[top].sum() / is_gap.sum()),
            "control_precision": float(is_control[top].mean()),
            "control_uplift": planted_uplift(is_control[top].mean(), is_control.mean()),
            "top_median_len_ratio": float(np.median(lengths[top]) / np.median(lengths)),
        }
        for t in TYPES:
            row[f"{t}_precision"] = float((types[top] == t).mean())
        out.append(row)
    return out


def recall_marks(order: np.ndarray, types: np.ndarray) -> dict[str, int]:
    # smallest k at which the cumulative share of gap needles reaches each mark
    cum = np.cumsum(np.isin(types[order], GAP)) / np.isin(types, GAP).sum()
    return {f"k_at_recall_{m}": int(np.argmax(cum >= m) + 1) for m in RECALL_MARKS}


def rf_cluster_order(ratio: np.ndarray, nearest: np.ndarray, labels: np.ndarray, min_members: int) -> np.ndarray:
    # exp37's cluster step: eligible clusters by descending median ratio, ties by lowest median nearest-mix cosine, small clusters last
    keys = []
    for c in np.unique(labels):
        members = np.where(labels == c)[0]
        eligible = len(members) >= min_members
        keys.append((not eligible, -float(np.median(ratio[members])), float(np.median(nearest[members])), c))
    order: list[int] = []
    for _, _, _, c in sorted(keys):
        members = np.where(labels == c)[0]
        order += members[np.argsort(-ratio[members], kind="stable")].tolist()
    return np.array(order)


def unique_mix(names: list[str], texts: dict[str, list[str]], emb_dir: Path) -> tuple[np.ndarray, list[str]]:
    # exp37's dedup on normalised text, in the given order of sets, indexing exp31's saved embeddings
    seen, norms, parts = set(), [], []
    for n in names:
        keep, nm_n = [], [normalise(t) for t in texts[n]]
        for i, nm in enumerate(nm_n):
            if nm not in seen:
                seen.add(nm)
                keep.append(i)
                norms.append(nm)
        parts.append(np.load(emb_dir / f"emb_{n}.npy", mmap_mode="r")[keep])
    return np.concatenate(parts).astype(np.float16), norms


def main():
    config = simple_parsing.parse(CurveConfig)
    missing = [k for k, v in asdict(config).items() if v == ""]
    assert not missing, f"required arguments not set: {missing}"
    run = Path(config.run_dir)
    assert not run.exists(), f"{run} exists; results/raw is append-only"
    setup_logging(run)
    set_seed(config.seed)
    write_json({"git_hash": get_git_hash(), **asdict(config)}, run / "config.json")
    cells = list(CELLS.values())

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
        rows = load_jsonl(Path(config.needles_dir) / f"needles_{t}.jsonl")
        banks[t] = load_pool(
            Path(config.needles_features) / f"needles_{t}.npz", Path(config.needles_dir) / f"needles_{t}.jsonl", cells
        )
        assert [r["id"] for r in rows] == list(np.load(Path(config.needles_features) / f"needles_{t}.npz")["ids"])
        banks[t]["source"] = np.array(["needle"] * len(rows))
        banks[t]["type"] = np.array([t] * len(rows))
    e1f, e1p = Path(config.exp01_dir) / "features", Path(config.exp01_pools)
    idfit = load_pool(e1f / "idfit.npz", e1p / "idfit.jsonl", cells)
    mix = concat([idfit, load_pool(e1f / "idmix.npz", e1p / "idmix.jsonl", cells)])
    dolma = load_pool(e1f / "dolma_full.npz", e1p / "dolma.jsonl", cells)
    LOGGER.info(f"deploy {len(deploy['len'])} rows, mix {len(mix['len'])}, dolma {len(dolma['len'])}")

    # activation and word scores, fixed per row so planted pools are assembled by indexing
    fixed: dict[str, dict[str, np.ndarray]] = {n: {} for n in ("deploy", *TYPES)}
    for name, cell in CELLS.items():
        scale = idfit[cell].std(0) + 1e-6
        knn_mix, knn_dolma = KNN(mix[cell], scale, config.knn_k), KNN(dolma[cell], scale, config.knn_k)
        for n, src in (("deploy", deploy), *banks.items()):
            fixed[n][f"dknn_{name}"] = knn_mix(src[cell]) - knn_dolma(src[cell])
    tfidf = tfidf_mix_vs_dolma(
        idfit["text"], [" ".join(t.split()[: config.dolma_tfidf_words]) for t in dolma["text"]], config.seed
    )
    for n, src in (("deploy", deploy), *banks.items()):
        fixed[n]["tfidf"] = tfidf(src["text"])
    LOGGER.info("activation and word scores done")

    # sentence embeddings: exp31's for the pool, fresh for the banks, with the same 1,000-character cut
    embedder = SentenceTransformer(EMBED_MODEL, device="cuda", revision=EMBED_REVISION)

    def encode(texts: list[str]) -> np.ndarray:
        return embedder.encode(
            [t[: config.embed_chars] for t in texts],
            batch_size=256,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )

    rng = np.random.default_rng(config.seed)
    emb = {"deploy": np.load(config.traffic_emb).astype(np.float32)}
    assert len(emb["deploy"]) == len(deploy["len"]), "saved pool embeddings do not match the pool"
    for t in TYPES:
        emb[t] = encode(banks[t]["text"])
    check_rows = rng.choice(len(deploy["len"]), size=config.n_embed_check, replace=False)
    pool_check = float(np.min(np.sum(encode([deploy["text"][i] for i in check_rows]) * emb["deploy"][check_rows], 1)))
    LOGGER.info(f"pool embedding check: min cosine {pool_check:.4f}")

    sft = pd.read_parquet(config.sft_path, columns=["first_user"])
    sft_texts = sft["first_user"].dropna().tolist()
    dpo_texts = load_dolci_dpo()["first_user"].tolist()
    emb_dir = Path(config.emb_v1)
    assert len(np.load(emb_dir / "emb_sft.npy", mmap_mode="r")) == len(sft_texts), "sft embeddings misaligned"
    assert len(np.load(emb_dir / "emb_dpo.npy", mmap_mode="r")) == len(dpo_texts), "dpo embeddings misaligned"
    mix_checks = {}
    for n, texts in (("sft", sft_texts), ("dpo", dpo_texts)):
        pick = rng.choice(len(texts), size=20, replace=False)
        saved = np.load(emb_dir / f"emb_{n}.npy", mmap_mode="r")[pick].astype(np.float32)
        mix_checks[n] = float(np.min(np.sum(encode([texts[i] for i in pick]) * saved, 1)))
    LOGGER.info(f"mix embedding checks: {mix_checks}")
    mixes = {
        "dpo": unique_mix(["dpo"], {"dpo": dpo_texts}, emb_dir),
        "full": unique_mix(["sft", "dpo"], {"sft": sft_texts, "dpo": dpo_texts}, emb_dir),
    }
    mix_sizes = {n: len(v[1]) for n, v in mixes.items()}
    LOGGER.info(f"mix sizes {mix_sizes}")
    mix_gpu = {
        n: (
            torch.from_numpy(e).cuda(),
            torch.from_numpy(np.array([text_hash(x) for x in norms], dtype=np.int64)).cuda(),
        )
        for n, (e, norms) in mixes.items()
    }
    del sft, sft_texts

    def planted(s: int) -> tuple[dict, np.ndarray]:
        parts = [("deploy", np.arange(len(deploy["len"])))] + [
            (t, np.arange(s * config.n_per_type, (s + 1) * config.n_per_type)) for t in TYPES
        ]
        srcs = {"deploy": deploy, **banks}
        pool = concat([take(srcs[n], idx) for n, idx in parts])
        for arm in fixed["deploy"]:
            pool[arm] = np.concatenate([fixed[n][arm][idx] for n, idx in parts])
        e = np.concatenate([emb[n][idx] for n, idx in parts]).astype(np.float32)
        assert len(pool["len"]) == len(deploy["len"]) + len(TYPES) * config.n_per_type
        return pool, e

    curve, marks, top10, scores0 = [], [], {}, None
    for s in config.needle_seeds:
        pool, e = planted(s)
        n = len(pool["len"])
        norms = [normalise(t) for t in pool["text"]]
        hashes = np.array([text_hash(x) for x in norms], dtype=np.int64)
        t_emb, t_hash = torch.from_numpy(e.astype(np.float16)).cuda(), torch.from_numpy(hashes).cuda()
        scores: dict[str, np.ndarray] = {
            "random": np.random.default_rng(1000 + s).random(n),
            "word_pct20": length_controlled(pool["tfidf"], pool["len"]),
            "act_cmean_pct20": length_controlled(pool["dknn_cmean"], pool["len"]),
            "act_cmax_pct20": length_controlled(pool["dknn_cmax"], pool["len"]),
        }
        nearest = {}
        for m, (m_emb, m_hash) in mix_gpu.items():
            ratio, near = density_ratios_batched(
                e.astype(np.float16),
                hashes,
                np.arange(n),
                t_emb,
                t_hash,
                m_emb,
                m_hash,
                k=config.rf_k,
                return_nearest_mix=True,
            )
            scores[f"rf_{m}_raw"], nearest[m] = ratio, near
            scores[f"rf_{m}_pct20"] = length_controlled(ratio, pool["len"])
        orders = {arm: jitter_order(sc, s) for arm, sc in scores.items()}
        act_labels = kmeans_labels(pool[CELLS["cmean"]], config.kmeans_k, config.seed, config.kmeans_n_init)
        orders["act_cmean_cluster"] = cluster_order(scores["act_cmean_pct20"], act_labels, n)
        emb_labels = KMeans(
            n_clusters=config.kmeans_k, n_init=config.kmeans_n_init, random_state=config.seed
        ).fit_predict(e)
        for m in mix_gpu:
            orders[f"rf_{m}_cluster"] = rf_cluster_order(
                scores[f"rf_{m}_raw"], nearest[m], emb_labels, config.min_cluster
            )
        assert set(orders) == set(ARMS), set(orders) ^ set(ARMS)
        for arm in ARMS:
            assert len(set(orders[arm].tolist())) == n, arm
            curve += curve_rows(orders[arm], pool["type"], pool["len"], config.ks, arm=arm, needle_seed=s)
            marks.append({"arm": arm, "needle_seed": s, **recall_marks(orders[arm], pool["type"])})
        LOGGER.info(
            f"seed {s}: "
            + ", ".join(
                f"{arm} p@100 {next(r['gap_precision'] for r in curve if r['arm'] == arm and r['needle_seed'] == s and r['k'] == 100):.3f}"
                for arm in ARMS
            )
        )
        if s == config.needle_seeds[0]:
            for arm in ("rf_dpo_raw", "word_pct20"):
                shown = [i for i in orders[arm] if pool["type"][i] == "deploy"][:10]
                top10[arm] = [
                    {
                        "id": f"{arm}_top_{j:02d}",
                        "source": str(pool["source"][i]),
                        "n_tokens": int(pool["len"][i]),
                        "text": pool["text"][i][:300],
                    }
                    for j, i in enumerate(shown)
                ]
            scores0 = [
                {
                    "id": f"seed0_{i:05d}",
                    "type": str(pool["type"][i]),
                    "source": str(pool["source"][i]),
                    "n_tokens": int(pool["len"][i]),
                    **{arm: round(float(sc[i]), 4) for arm, sc in scores.items()},
                    **{f"rank_{arm}": int(np.where(orders[arm] == i)[0][0]) for arm in ARMS},
                }
                for i in range(n)
            ]
    write_json(curve, run / "curve.json")
    write_json(marks, run / "recall_marks.json")
    write_json(top10, run / "top10_nonneedle_seed0.json")
    write_jsonl(scores0, run / "scores_seed0.jsonl")

    def mean_at(arm: str, key: str, k: int) -> float:
        vals = [r[key] for r in curve if r["arm"] == arm and r["k"] == k]
        assert len(vals) == len(config.needle_seeds)
        return float(np.mean(vals))

    def list_at(arm: str, key: str, k: int) -> list[float]:
        return [r[key] for r in curve if r["arm"] == arm and r["k"] == k]

    def balanced(arm: str, k: int) -> bool:
        return all(0.67 <= r <= 1.5 for r in list_at(arm, "top_median_len_ratio", k))

    def b1(arm: str) -> dict:
        gap, control = mean_at(arm, "gap_uplift", 100), mean_at(arm, "control_uplift", 100)
        if gap >= 5 and control <= 2 and balanced(arm, 100):
            verdict = "passes"
        elif gap < 2:
            verdict = "fails"
        else:
            verdict = "ambiguous"
        return {
            "gap_uplift@100": gap,
            "control_uplift@100": control,
            "len_ratios@100": list_at(arm, "top_median_len_ratio", 100),
            "verdict": verdict,
        }

    def b2() -> dict:
        per_k = {}
        for k in (100, 500):
            rf, word = mean_at("rf_dpo_raw", "gap_uplift", k), mean_at("word_pct20", "gap_uplift", k)
            control = mean_at("rf_dpo_raw", "control_uplift", k)
            better = rf >= 1.5 * word and rf >= word + 2 and control <= 2 and balanced("rf_dpo_raw", k)
            worse = rf <= word / 1.5
            per_k[str(k)] = {
                "rf_dpo_raw": rf,
                "word_pct20": word,
                "rf_control_uplift": control,
                "better": better,
                "worse": worse,
            }
        if all(v["better"] for v in per_k.values()):
            verdict = "better"
        elif all(v["worse"] for v in per_k.values()):
            verdict = "worse"
        else:
            verdict = "equivalent"
        return {"per_k": per_k, "verdict": verdict}

    summary = {
        arm: {
            **{f"gap_precision@{k}": round(mean_at(arm, "gap_precision", k), 4) for k in config.ks},
            **{f"gap_recall@{k}": round(mean_at(arm, "gap_recall", k), 4) for k in config.ks},
            "k_at_recall_0.5": [m["k_at_recall_0.5"] for m in marks if m["arm"] == arm],
        }
        for arm in ARMS
    }
    random_uplift = mean_at("random", "gap_uplift", 100)
    decision = {
        "git_hash": get_git_hash(),
        "b1_planted_check": {
            arm: b1(arm) for arm in ("rf_dpo_raw", "rf_dpo_cluster", "rf_full_raw", "rf_full_cluster")
        },
        "b2_rf_dpo_raw_vs_word_pct20": b2(),
        "summary_mean_over_seeds": summary,
        "sanity": {
            "rows_per_seed": int(len(deploy["len"]) + len(TYPES) * config.n_per_type),
            "random_gap_uplift@100": random_uplift,
            "random_in_band": 0 <= random_uplift <= 3,
            "pool_embedding_min_cosine": pool_check,
            "mix_embedding_min_cosine": mix_checks,
            "mix_sizes": mix_sizes,
        },
    }
    write_json(decision, run / "decision.json")
    print(
        "B1 planted check: "
        + "; ".join(
            f"{a} {v['verdict']} (gap uplift@100 {v['gap_uplift@100']:.2f})"
            for a, v in decision["b1_planted_check"].items()
        )
    )
    print(f"B2 rf_dpo_raw vs word_pct20: {decision['b2_rf_dpo_raw_vs_word_pct20']['verdict']}")
    for arm in ARMS:
        print(
            f"{arm}: "
            + " ".join(f"p@{k} {summary[arm][f'gap_precision@{k}']:.3f}" for k in config.ks)
            + f" k50 {summary[arm]['k_at_recall_0.5']}"
        )
    print(f"sanity {decision['sanity']}")
    print(f"decision written to {run / 'decision.json'}")


if __name__ == "__main__":
    main()
