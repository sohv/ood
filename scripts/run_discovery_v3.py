# activations_vs_words: the iterative_discovery gap loop with an activation arm, a tf-idf word arm and a random arm, counting gaps only activations find.
# uv run -m scripts.run_discovery_v3 --run_dir results/raw/261019_activations_vs_words_act_vs_word_v1 --repro_two_sided_dir results/raw/261018_repro_two_sided_repro_v1 --iterative_discovery_dir results/raw/261019_iterative_discovery_v2 --repro_two_sided_pools_dir data/processed/repro_two_sided --model_id allenai/Olmo-3-7B-Instruct-DPO --seed 0

import os

os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "8")

import asyncio
import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import simple_parsing
import torch
from openai import AsyncOpenAI
from sentence_transformers import SentenceTransformer

from src.data.io import load_jsonl
from src.data.pools import load_dolci_dpo
from src.interp.features import LAYERS
from src.metrics.density_ratio import length_controlled
from src.metrics.detectors import KNN, tfidf_mix_vs_dolma
from src.metrics.discovery import (
    NAMING_PROMPT_GRAIN,
    centroid,
    cluster,
    match_gap,
    name_clusters,
    naming_prompt,
    parse_naming,
)
from src.metrics.io import write_json, write_jsonl
from src.utils.git import get_git_hash
from src.utils.logging import setup_logging
from src.utils.seed import set_seed

LOGGER = logging.getLogger(__name__)
ARMS = ("act", "word", "random")


@dataclass
class DiscoveryV3Config:
    run_dir: str = ""
    repro_two_sided_dir: str = ""
    iterative_discovery_dir: str = ""
    repro_two_sided_pools_dir: str = ""
    model_id: str = "allenai/Olmo-3-7B-Instruct-DPO"
    seed: int = 0
    cluster_seeds: list[int] = field(default_factory=lambda: [0, 1, 2])
    layer: int = 16
    rounds: int = 5
    top_k: int = 500
    n_clusters: int = 10
    n_shown: int = 25
    min_size: int = 15
    same_gap_cos: float = 0.85
    expected_cut: float = 0.6003
    namer: str = "anthropic/claude-sonnet-4.6"
    # stops after the pipeline check and the reviewed rows, before any naming call
    dry_run: bool = False


def rank_rounds(score: np.ndarray, lengths: np.ndarray, rounds: int, top_k: int) -> list[np.ndarray]:
    # iterative_discovery v2's loop without reference updates: top_k of the unreviewed rows per round, 20-bin length control
    remaining = np.ones(len(score), dtype=bool)
    out = []
    for rnd in range(1, rounds + 1):
        idx = np.where(remaining)[0]
        lc = length_controlled(score[idx], lengths[idx])
        top = idx[np.argsort(-(lc + np.random.default_rng(rnd).random(len(lc)) * 1e-9))[:top_k]]
        remaining[top] = False
        out.append(top)
    return out


def random_rounds(n: int, rounds: int, top_k: int) -> list[np.ndarray]:
    remaining = np.ones(n, dtype=bool)
    out = []
    for rnd in range(1, rounds + 1):
        pick = np.random.default_rng(10_000 + rnd).choice(np.where(remaining)[0], size=top_k, replace=False)
        remaining[pick] = False
        out.append(pick)
    return out


def main():
    config = simple_parsing.parse(DiscoveryV3Config)
    assert (
        config.run_dir
        and config.repro_two_sided_dir
        and config.iterative_discovery_dir
        and config.repro_two_sided_pools_dir
    ), "input paths required"
    run = Path(config.run_dir)
    assert not run.exists(), f"{run} exists; results/raw is append-only"
    setup_logging(run)
    set_seed(config.seed)
    c9 = json.loads((Path(config.iterative_discovery_dir) / "config.json").read_text())
    write_json({"git_hash": get_git_hash(), **asdict(config), "iterative_discovery_config": c9}, run / "config.json")
    li = LAYERS.index(config.layer)

    wc, lm = load_jsonl(c9["wildchat_pool"]), load_jsonl(c9["lmsys_pool"])
    wf, lf = np.load(c9["wildchat_features"]), np.load(c9["lmsys_features"])
    assert list(wf["ids"]) == [r["id"] for r in wc] and list(lf["ids"]) == [r["id"] for r in lm], "features misaligned"
    texts = [r["text"] for r in wc] + [r["text"] for r in lm]
    source = np.array(["wildchat"] * len(wc) + ["lmsys"] * len(lm))
    lengths = np.concatenate([wf["n_content_tokens"], lf["n_content_tokens"]])
    e1 = Path(config.repro_two_sided_dir) / "features"

    def feats(pooling: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        x = np.vstack([wf[pooling][:, li], lf[pooling][:, li]]).astype(np.float32)
        mix = np.vstack([np.load(e1 / f"{n}.npz")[pooling][:, li] for n in ("idfit", "idmix")]).astype(np.float32)
        dolma = np.load(e1 / "dolma_full.npz")[pooling][:, li].astype(np.float32)
        scale = np.load(e1 / "idfit.npz")[pooling][:, li].astype(np.float32).std(0) + 1e-6
        return x, mix, dolma, scale

    def dknn(pooling: str) -> np.ndarray:
        x, mix, dolma, scale = feats(pooling)
        return KNN(mix, scale)(x) - KNN(dolma, scale)(x)

    # pipeline check: iterative_discovery v2's round-1 dknn top 500 (cmean) must come back exactly
    stored = {r["id"]: r["rows"] for r in load_jsonl(Path(config.iterative_discovery_dir) / "reviews.jsonl")}
    check = rank_rounds(dknn("cmean"), lengths, 1, config.top_k)[0]
    assert check.tolist() == stored["dknn_top_r1"], "recomputed iterative_discovery v2 round-1 ranking differs"
    print("pipeline check: iterative_discovery v2 dknn round-1 top 500 reproduced exactly")

    repro_two_sided_pools = Path(config.repro_two_sided_pools_dir)
    tfidf = tfidf_mix_vs_dolma(
        [r["text"] for r in load_jsonl(repro_two_sided_pools / "idfit.jsonl")],
        [r["text"] for r in load_jsonl(repro_two_sided_pools / "dolma.jsonl")],
        config.seed,
    )
    reviewed = {
        "act": rank_rounds(dknn("cmax"), lengths, config.rounds, config.top_k),
        "word": rank_rounds(tfidf(texts), lengths, config.rounds, config.top_k),
        "random": random_rounds(len(texts), config.rounds, config.top_k),
    }
    for arm, rs in reviewed.items():
        assert len(np.unique(np.concatenate(rs))) == config.rounds * config.top_k, f"{arm} reviewed rows repeat"
    overlap = len(np.intersect1d(np.concatenate(reviewed["act"]), np.concatenate(reviewed["word"])))
    print(f"reviewed rows per arm {config.rounds * config.top_k}, act-word overlap {overlap}")
    if config.dry_run:
        return

    embedder = SentenceTransformer(c9["embed_model"], device="cuda", revision=c9["embed_model_revision"])

    def encode(t: list[str]) -> np.ndarray:
        return embedder.encode(
            t, batch_size=256, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False
        )

    emb = encode(texts)
    dolci = load_dolci_dpo()["first_user"].drop_duplicates().tolist()
    ref = torch.from_numpy(encode(dolci)).cuda()
    coverage = np.concatenate(
        [
            (torch.from_numpy(emb[i : i + 4096]).cuda() @ ref.T).max(dim=1).values.cpu().numpy()
            for i in range(0, len(emb), 4096)
        ]
    )
    cut = float(np.percentile(coverage, 25))
    assert abs(cut - config.expected_cut) < 0.001, f"coverage cut {cut:.4f} differs from iterative_discovery v2's"
    pool_median_len = float(np.median(lengths))

    client = AsyncOpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.environ["OPENROUTER_API_KEY"])
    clusters, per_seed = [], {}
    for s in config.cluster_seeds:
        groups = []
        for arm in ARMS:
            for rnd, rows in enumerate(reviewed[arm], start=1):
                labels = cluster(emb[rows], config.n_clusters, s)
                groups += [(arm, rnd, c, rows[labels == c]) for c in range(config.n_clusters)]
        prompts = [
            naming_prompt([texts[i] for i in m], config.n_shown, config.seed, template=NAMING_PROMPT_GRAIN)
            for *_, m in groups
        ]
        replies = asyncio.run(name_clusters(client, config.namer, prompts, str(run / "naming_cache")))
        gaps = {arm: [] for arm in ARMS}
        for (arm, rnd, c, m), reply in zip(groups, replies):
            named = parse_naming(reply)
            n_shown = min(config.n_shown, len(m))
            coherent = None if named["shared"] is None else named["shared"] >= n_shown / 2
            med_cov = float(np.median(coverage[m]))
            row = {
                "id": f"s{s}_{arm}_r{rnd}_c{c}",
                "seed": s,
                "arm": arm,
                "round": rnd,
                "cluster": c,
                "size": len(m),
                **named,
                "coherent": coherent,
                "median_coverage": med_cov,
                "confirmed": bool(coherent) and len(m) >= config.min_size and med_cov < cut,
                "lmsys_share": float(np.mean(source[m] == "lmsys")),
                "median_len_ratio": float(np.median(lengths[m]) / pool_median_len),
                "examples": [" ".join(texts[i].split())[:200] for i in m[:5]],
            }
            if row["confirmed"]:
                cen = centroid(emb[m])
                j = match_gap(cen, [g["centroid"] for g in gaps[arm]], config.same_gap_cos)
                row["gap_index"] = len(gaps[arm]) if j is None else j
                if j is None:
                    gaps[arm].append({"centroid": cen, "first_cluster": row["id"], "name": row["name"], "members": m})
            clusters.append(row)

        def unique(gaps: dict, a: str, b: str) -> list[dict]:
            return [
                g
                for g in gaps[a]
                if match_gap(g["centroid"], [h["centroid"] for h in gaps[b]], config.same_gap_cos) is None
            ]

        act_u, word_u = unique(gaps, "act", "word"), unique(gaps, "word", "act")
        n_act, n_rand = len(gaps["act"]), len(gaps["random"])
        if len(act_u) >= 2 and n_act >= 2 * n_rand and n_act >= n_rand + 2:
            outcome = "adds_something"
        elif len(act_u) == 0 or n_act <= n_rand + 1:
            outcome = "adds_nothing"
        else:
            outcome = "ambiguous"
        per_seed[s] = {
            "distinct_gaps": {arm: len(gaps[arm]) for arm in ARMS},
            "act_unique": len(act_u),
            "word_unique": len(word_u),
            "outcome": outcome,
            "null_names": sum(r["coherent"] is None for r in clusters if r["seed"] == s),
            "unique_gaps": {
                label: [
                    {
                        "name": g["name"],
                        "first_cluster": g["first_cluster"],
                        "size": len(g["members"]),
                        "examples": [" ".join(texts[i].split())[:200] for i in g["members"][:5]],
                        "nearest_dolci": [
                            " ".join(dolci[k].split())[:200]
                            for k in (ref @ torch.from_numpy(g["centroid"]).cuda()).topk(3).indices.tolist()
                        ],
                    }
                    for g in gs
                ]
                for label, gs in (("act", act_u), ("word", word_u))
            },
            "all_gap_names": {arm: [g["name"] for g in gaps[arm]] for arm in ARMS},
        }
        LOGGER.info(
            f"seed {s}: {per_seed[s]['distinct_gaps']}, act_unique {len(act_u)}, word_unique {len(word_u)}, {outcome}"
        )

    outcomes = [v["outcome"] for v in per_seed.values()]
    verdict = next((o for o in ("adds_something", "adds_nothing") if outcomes.count(o) >= 2), "ambiguous")
    act_rows, word_rows = np.concatenate(reviewed["act"]), np.concatenate(reviewed["word"])
    per_round = {
        f"s{s}_{arm}_r{rnd}": {
            k: sum(test(r) for r in clusters if r["seed"] == s and r["arm"] == arm and r["round"] == rnd)
            for k, test in (
                ("coherent", lambda r: bool(r["coherent"])),
                ("min_size", lambda r: r["size"] >= config.min_size),
                ("below_cut", lambda r: r["median_coverage"] < cut),
                ("confirmed", lambda r: r["confirmed"]),
            )
        }
        for s in config.cluster_seeds
        for arm in ARMS
        for rnd in range(1, config.rounds + 1)
    }
    decision = {
        "verdict": verdict,
        "per_seed": {str(s): {k: v for k, v in d.items() if k != "unique_gaps"} for s, d in per_seed.items()},
        "coverage_cut": cut,
        "reviewed_overlap_act_word": len(np.intersect1d(act_rows, word_rows)),
        "reviewed_top_median_len_ratio": {
            arm: float(np.median(lengths[np.concatenate(rs)]) / pool_median_len) for arm, rs in reviewed.items()
        },
        "reviewed_lmsys_share": {
            arm: float(np.mean(source[np.concatenate(rs)] == "lmsys")) for arm, rs in reviewed.items()
        },
        "per_round": per_round,
    }
    write_json(decision, run / "decision.json")
    write_json({str(s): d["unique_gaps"] for s, d in per_seed.items()}, run / "unique_gaps.json")
    write_jsonl(clusters, run / "clusters.jsonl")
    write_jsonl(
        [
            {"id": f"{arm}_r{i + 1}", "arm": arm, "round": i + 1, "rows": rs.tolist()}
            for arm in ARMS
            for i, rs in enumerate(reviewed[arm])
        ],
        run / "reviews.jsonl",
    )
    for s, d in per_seed.items():
        print(
            f"seed {s}: distinct {d['distinct_gaps']}, act-unique {d['act_unique']}, word-unique {d['word_unique']}, {d['outcome']}"
        )
    print(f"{verdict=}")
    print(f"decision written to {run / 'decision.json'}")


if __name__ == "__main__":
    main()
