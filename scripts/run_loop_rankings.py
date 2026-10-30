# loop_three_finders: deep_gap_loop's skip loop driven by three rankings of pipeline_million's topics (neighbour count, tf-idf word score, flash rater), distinct gaps in 200 reviews each.
# uv run -m scripts.run_loop_rankings --run_dir results/raw/261027_loop_three_finders_loop_rankings_v1 --pipeline_million_dir results/raw/261021_pipeline_million_gap_pipeline_v2 --pipeline_million_pools data/processed/pipeline_million_v2 --deep_gap_loop_dir results/raw/261022_deep_gap_loop_v1 --rater_path results/raw/261026_auto_rater_finder_autorater_v1/scores_flash_B_full.jsonl --rounds 20 --seed 0

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("OMP_NUM_THREADS", "96")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "96")

import asyncio
import json
import logging
import resource
import shutil
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import simple_parsing
import torch
from openai import AsyncOpenAI

from src.data.gap_pool import load_mix, load_traffic_pool
from src.data.io import load_jsonl
from src.metrics.discovery import NAMING_PROMPT_GRAIN, centroid, name_clusters, naming_prompt, parse_naming
from src.metrics.gap_pipeline import text_hash
from src.metrics.io import write_json, write_jsonl
from src.metrics.two_sided_knn import word_logits
from src.utils.git import get_git_hash
from src.utils.logging import setup_logging
from src.utils.seed import set_seed

LOGGER = logging.getLogger(__name__)
ARMS = ("A", "B", "C")


@dataclass
class LoopRankingsConfig:
    run_dir: str = ""
    pipeline_million_dir: str = ""
    pipeline_million_pools: str = ""
    deep_gap_loop_dir: str = ""
    rater_path: str = ""
    namer_model: str = "anthropic/claude-sonnet-4.6"
    seed: int = 0
    rounds: int = 20
    per_round: int = 10
    n_named: int = 25
    shared_needed: int = 13
    ratio_multiple: float = 3.0
    n_random_subsets: int = 1000
    subset_size: int = 500
    same_gap: float = 0.85
    theme_cos: float = 0.70
    template_chars: int = 60
    template_share: float = 0.3
    line_rank: int = 293
    margin: int = 10
    jitter: float = 1e-3
    n_examples: int = 10
    example_chars: int = 400
    mem_wait_gb: float = 70.0
    mem_poll_s: int = 180


def main():
    config = simple_parsing.parse(LoopRankingsConfig)
    assert (
        config.run_dir
        and config.pipeline_million_dir
        and config.pipeline_million_pools
        and config.deep_gap_loop_dir
        and config.rater_path
    )
    run = Path(config.run_dir)
    assert not run.exists(), f"{run} exists; results/raw is append-only"
    setup_logging(run)
    set_seed(config.seed)
    torch.set_num_threads(int(os.environ["OMP_NUM_THREADS"]))
    write_json({"git_hash": get_git_hash(), **asdict(config)}, run / "config.json")
    rng = np.random.default_rng(config.seed)
    shutil.copytree(Path(config.deep_gap_loop_dir) / "naming_cache", run / "naming_cache")
    cache = run / "naming_cache"
    n_cache_start = len(list(cache.glob("*.json")))

    # the gpu belongs to fix_loop_round2, so the new lmsys rows are embedded on cpu
    c37 = json.loads((Path(config.pipeline_million_dir) / "config.json").read_text())
    rows, t_np, _, _ = load_traffic_pool(
        c37["iterative_discovery_dir"],
        c37["self_nature_a_dir"],
        c37["emb_v2"],
        c37["new_lmsys_shards"],
        c37["embed_chars"],
        device="cpu",
    )
    labels = np.load(Path(config.pipeline_million_dir) / "labels.npy")
    ratio = np.load(Path(config.pipeline_million_dir) / "ratio.npy")
    assert len(labels) == len(rows) == len(ratio), "pool does not match pipeline_million's labels"
    for p in load_jsonl(Path(config.pipeline_million_pools) / "prompts.jsonl"):
        assert rows[p["row"]]["text"] == p["text"], f"row {p['row']} differs from pipeline_million"
    clusters = {c["id"]: c for c in load_jsonl(Path(config.pipeline_million_dir) / "clusters.jsonl")}
    order37 = [int(c["id"][1:]) for c in sorted((c for c in clusters.values() if c["rank"]), key=lambda c: c["rank"])]
    LOGGER.info(f"pool {len(rows)} rows, {len(order37)} ranked topics")

    emb = t_np.astype(np.float32)
    members = {c: np.where(labels == c)[0] for c in order37}
    cent = {c: centroid(emb[members[c]]) for c in order37}
    sig: dict[int, str | None] = {}
    for c in order37:
        prefix, count = Counter(rows[i]["norm"][: config.template_chars] for i in members[c]).most_common(1)[0]
        sig[c] = prefix if count >= config.template_share * len(members[c]) else None

    # arm a: deep_gap_loop's ratio and line, rng drawn first exactly as deep_gap_loop did
    subset_medians = [
        float(np.median(ratio[rng.choice(len(rows), config.subset_size, replace=False)]))
        for _ in range(config.n_random_subsets)
    ]
    ratio_line = config.ratio_multiple * float(np.median(subset_medians))
    d41 = json.loads((Path(config.deep_gap_loop_dir) / "decision.json").read_text())
    assert round(ratio_line, 4) == d41["ratio_line"], f"{ratio_line=} differs from deep_gap_loop's"
    score = {"A": {c: float(clusters[f"c{c}"]["ratio"]) for c in order37}}

    # arm b: count_vs_tfidf's two-fold tf-idf logistic score on the full mix, topic median
    m_hash, m_text = load_mix(c37["sft_path"], c37["emb_v1"], with_text=True)[1:]
    t_hash = np.array([text_hash(r["norm"]) for r in rows], dtype=np.int64)
    # the container is capped at 117 gb, so tf-idf waits until other jobs leave room
    while (mem_gb := int(Path("/sys/fs/cgroup/memory.current").read_text()) / 1e9) > config.mem_wait_gb:
        LOGGER.warning(f"container memory {mem_gb:.1f} gb above {config.mem_wait_gb} gb, waiting before tf-idf")
        time.sleep(config.mem_poll_s)
    LOGGER.info(f"container memory {mem_gb:.1f} gb, starting tf-idf")
    logit = word_logits(
        [r["text"] for r in rows],
        m_text,
        (t_hash & 1).astype(np.int64),
        (m_hash & 1).astype(np.int64),
        seed=config.seed,
    )
    del m_text
    score["B"] = {c: float(np.median(logit[members[c]])) for c in order37}
    LOGGER.info("word score done")

    # arm c: auto_rater_finder's flash rater, ties broken by seeded jitter
    rater = {int(r["id"][1:]): float(r["gap_score"]) for r in load_jsonl(config.rater_path)}
    assert set(rater) == set(order37), "rater must score every ranked topic"
    jitter = np.random.default_rng(config.seed).uniform(0, config.jitter, len(order37))
    score["C"] = {c: rater[c] + float(j) for c, j in zip(order37, jitter, strict=True)}

    order = {"A": order37, **{a: sorted(order37, key=lambda c: -score[a][c]) for a in ("B", "C")}}
    line = {"A": ratio_line, **{a: score[a][order[a][config.line_rank - 1]] for a in ("B", "C")}}
    over_line = {a: sum(v >= line[a] for v in score[a].values()) for a in ARMS}
    LOGGER.info(f"lines {line}, topics over line {over_line}")

    def same(a: int, b: int, cos: float) -> bool:
        return float(cent[a] @ cent[b]) >= cos or (sig[a] is not None and sig[a] == sig[b])

    client = AsyncOpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.environ["OPENROUTER_API_KEY"])
    names: dict[int, dict] = {}

    def review(batch: list[int]) -> None:
        todo = [c for c in batch if c not in names]
        prompts = [
            naming_prompt(
                [rows[i]["text"] for i in members[c]], config.n_named, config.seed, template=NAMING_PROMPT_GRAIN
            )
            for c in todo
        ]
        replies = asyncio.run(name_clusters(client, config.namer_model, prompts, str(cache)))
        for c, rep in zip(todo, replies, strict=True):
            names[c] = parse_naming(rep)

    def confirmed(arm: str, c: int) -> bool:
        shared = names[c].get("shared")
        return shared is not None and shared >= config.shared_needed and score[arm][c] >= line[arm]

    def merge(arm: str, reviewed: list[int], cos: float) -> list[list[int]]:
        gaps: list[list[int]] = []
        for c in reviewed:
            if not confirmed(arm, c):
                continue
            hit = next((g for g in gaps if same(c, g[0], cos)), None)
            if hit is None:
                gaps.append([c])
            else:
                hit.append(c)
        return gaps

    def loop(arm: str) -> tuple[list[int], list[dict]]:
        remaining, reviewed, per_round = list(order[arm]), [], []
        for rnd in range(1, config.rounds + 1):
            batch = remaining[: config.per_round]
            review(batch)
            reviewed += batch
            new_conf = [c for c in batch if confirmed(arm, c)]
            rest = [c for c in remaining if c not in batch]
            dropped = [c for c in rest if any(same(c, g, config.same_gap) for g in new_conf)]
            remaining = [c for c in rest if c not in dropped]
            per_round.append(
                {
                    "id": f"round_{rnd}",
                    "reviewed": batch,
                    "confirmed": new_conf,
                    "dropped_as_same_gap": dropped,
                    "distinct_so_far": len(merge(arm, reviewed, config.same_gap)),
                }
            )
            LOGGER.info(
                f"arm {arm} round {rnd}: {len(new_conf)} confirmed, {per_round[-1]['distinct_so_far']} distinct"
            )
        return reviewed, per_round

    reviewed, rounds_log, calls = {}, {}, {}
    for arm in ARMS:
        before = len(list(cache.glob("*.json")))
        reviewed[arm], rounds_log[arm] = loop(arm)
        calls[arm] = len(list(cache.glob("*.json"))) - before

    gaps = {a: merge(a, reviewed[a], config.same_gap) for a in ARMS}
    themes = {a: merge(a, reviewed[a], config.theme_cos) for a in ARMS}
    n = {a: len(g) for a, g in gaps.items()}
    # recorded, not asserted: cpu embeddings of the new rows can move a centroid cosine across 0.85
    reviewed41 = [c for r in d41["loop_rounds"] for c in r["reviewed"]]
    a_check = {
        "distinct": n["A"],
        "deep_gap_loop_distinct": d41["distinct_gaps"]["loop"],
        "equals_deep_gap_loop_distinct": n["A"] == d41["distinct_gaps"]["loop"],
        "equals_deep_gap_loop_reviewed": reviewed["A"] == reviewed41,
        "reviewed_only_in_A": [c for c in reviewed["A"] if c not in reviewed41],
        "reviewed_only_in_deep_gap_loop": [c for c in reviewed41 if c not in reviewed["A"]],
    }
    if not (a_check["equals_deep_gap_loop_distinct"] and a_check["equals_deep_gap_loop_reviewed"]):
        LOGGER.warning(f"arm A does not reproduce deep_gap_loop: {a_check}")
    conf = {a: [c for c in reviewed[a] if confirmed(a, c)] for a in ARMS}

    def verdict(arm: str) -> str:
        if n["A"] >= n[arm] + config.margin:
            return "A_beats_arm"
        if n[arm] >= n["A"] + config.margin:
            return "arm_beats_A"
        return "tie"

    def overlap(arm: str) -> dict:
        shared = set(conf[arm]) & set(conf["A"])
        near = [c for c in conf[arm] if any(same(c, g, config.same_gap) for g in conf["A"])]
        return {
            "confirmed_in_both": len(shared),
            "share_of_arm_confirmed": len(shared) / max(len(conf[arm]), 1),
            "arm_confirmed_same_gap_as_an_A_confirmed": len(near),
            "reviewed_in_both": len(set(reviewed[arm]) & set(reviewed["A"])),
        }

    def describe(arm: str, c: int) -> dict:
        return {
            "topic": c,
            "rank": order[arm].index(c) + 1,
            "score": score[arm][c],
            "pipeline_million_ratio": score["A"][c],
            "size": len(members[c]),
            **{k: names[c].get(k) for k in ("name", "description", "shared")},
            "confirmed": confirmed(arm, c),
        }

    decision = {
        "distinct_gaps": {**n, "random": d41["distinct_gaps"]["random"]},
        "A_reproduces_deep_gap_loop": a_check,
        "verdict": {a: verdict(a) for a in ("B", "C")},
        "themes_0.70": {a: len(t) for a, t in themes.items()},
        "confirmed_topics": {a: len(c) for a, c in conf.items()},
        "overlap_with_A": {a: overlap(a) for a in ("B", "C")},
        "median_pipeline_million_ratio_of_confirmed": {
            a: float(np.median([score["A"][c] for c in conf[a]])) for a in ARMS
        },
        "lines": line,
        "rater_raw_line": rater[order["C"][config.line_rank - 1]],
        "topics_over_line": over_line,
        "namer_calls_new": calls,
        "namer_calls_total_new": len(list(cache.glob("*.json"))) - n_cache_start,
        "topics_named": len(names),
        "null_names": {a: sum(names[c].get("shared") is None for c in reviewed[a]) for a in ARMS},
        "null_names_total": sum(v.get("shared") is None for v in names.values()),
        "peak_rss_gb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6,
        "loop_rounds": rounds_log,
        "gaps": {a: [[describe(a, c) for c in g] for g in gl] for a, gl in gaps.items()},
    }
    write_json(decision, run / "decision.json")
    for arm in ARMS:
        write_jsonl(
            [
                {
                    "id": f"{arm}_c{c}",
                    "arm": arm,
                    **describe(arm, c),
                    "examples": [
                        rows[i]["text"][: config.example_chars]
                        for i in np.random.default_rng(config.seed).choice(
                            members[c], size=min(config.n_examples, len(members[c])), replace=False
                        )
                    ],
                }
                for c in reviewed[arm]
            ],
            run / f"reviewed_topics_{arm}.jsonl",
        )
    for a in ARMS:
        print(f"arm {a}: {n[a]} distinct gaps, {len(themes[a])} themes, {len(conf[a])} confirmed, line {line[a]:.4f}")
    print(f"random (deep_gap_loop): {d41['distinct_gaps']['random']}; verdicts {decision['verdict']}")
    print(f"decision written to {run / 'decision.json'}")


if __name__ == "__main__":
    main()
