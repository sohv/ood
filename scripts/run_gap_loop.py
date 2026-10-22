# exp39/exp41: iterative gap discovery on exp37's clusters (loop that removes same-gap topics) against walking down the ranking and random topics.
# uv run -m scripts.run_gap_loop --run_dir results/raw/261022_exp41_gap_loop_v1 --exp37_dir results/raw/261021_exp37_gap_pipeline_v2 --exp37_pools data/processed/exp37_v2 --rounds 20 --template_chars 60 --adds_margin 10 --nothing_margin 3 --seed 0

import asyncio
import json
import logging
import os
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import simple_parsing
from openai import AsyncOpenAI

from src.data.gap_pool import load_traffic_pool
from src.data.io import load_jsonl
from src.metrics.discovery import NAMING_PROMPT_GRAIN, centroid, name_clusters, naming_prompt, parse_naming
from src.metrics.io import write_json, write_jsonl
from src.utils.git import get_git_hash
from src.utils.logging import setup_logging
from src.utils.seed import set_seed

LOGGER = logging.getLogger(__name__)


@dataclass
class GapLoopConfig:
    run_dir: str = ""
    exp37_dir: str = ""
    exp37_pools: str = ""
    namer_model: str = "anthropic/claude-sonnet-4.6"
    seed: int = 0
    rounds: int = 5
    per_round: int = 10
    n_named: int = 25
    shared_needed: int = 13
    ratio_multiple: float = 3.0
    n_random_subsets: int = 1000
    subset_size: int = 500
    same_gap: float = 0.85
    # 0 turns template collapse off, as in exp39
    template_chars: int = 0
    template_share: float = 0.3
    adds_margin: int = 3
    nothing_margin: int = 1


def main():
    config = simple_parsing.parse(GapLoopConfig)
    assert config.run_dir and config.exp37_dir and config.exp37_pools, "run_dir, exp37_dir, exp37_pools required"
    run = Path(config.run_dir)
    assert not run.exists(), f"{run} exists; results/raw is append-only"
    setup_logging(run)
    set_seed(config.seed)
    write_json({"git_hash": get_git_hash(), **asdict(config)}, run / "config.json")
    rng = np.random.default_rng(config.seed)

    c37 = json.loads((Path(config.exp37_dir) / "config.json").read_text())
    rows, t_np, source, _ = load_traffic_pool(
        c37["exp09_dir"], c37["exp31_dir"], c37["emb_v2"], c37["new_lmsys_shards"], c37["embed_chars"]
    )
    labels = np.load(Path(config.exp37_dir) / "labels.npy")
    ratio = np.load(Path(config.exp37_dir) / "ratio.npy")
    assert len(labels) == len(rows) == len(ratio), "pool does not match exp37's labels"
    # rebuilt pool must match exp37 row by row on every prompt exp37 sampled
    for p in load_jsonl(Path(config.exp37_pools) / "prompts.jsonl"):
        assert rows[p["row"]]["text"] == p["text"], f"row {p['row']} differs from exp37"
    clusters = {c["id"]: c for c in load_jsonl(Path(config.exp37_dir) / "clusters.jsonl")}
    ranked = sorted((c for c in clusters.values() if c["rank"]), key=lambda c: c["rank"])
    order = [int(c["id"][1:]) for c in ranked]
    LOGGER.info(f"pool {len(rows)} rows, {len(order)} eligible clusters, exp37 ranking reused")

    emb = t_np.astype(np.float32)
    members = {c: np.where(labels == c)[0] for c in order}
    cent = {c: centroid(emb[members[c]]) for c in order}
    subset_medians = [
        float(np.median(ratio[rng.choice(len(rows), config.subset_size, replace=False)]))
        for _ in range(config.n_random_subsets)
    ]
    ratio_line = config.ratio_multiple * float(np.median(subset_medians))
    c_ratio = {c: float(clusters[f"c{c}"]["ratio"]) for c in order}
    sig: dict[int, str | None] = dict.fromkeys(order)
    if config.template_chars:
        for c in order:
            prefix, count = Counter(rows[i]["norm"][: config.template_chars] for i in members[c]).most_common(1)[0]
            sig[c] = prefix if count >= config.template_share * len(members[c]) else None
    LOGGER.info(f"{sum(s is not None for s in sig.values())} topics have a template signature")

    def same(a: int, b: int) -> bool:
        return float(cent[a] @ cent[b]) >= config.same_gap or (sig[a] is not None and sig[a] == sig[b])

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
        replies = asyncio.run(name_clusters(client, config.namer_model, prompts, str(run / "naming_cache")))
        for c, rep in zip(todo, replies, strict=True):
            names[c] = parse_naming(rep)

    def confirmed(c: int) -> bool:
        shared = names[c].get("shared")
        return shared is not None and shared >= config.shared_needed and c_ratio[c] >= ratio_line

    def distinct(reviewed: list[int]) -> list[list[int]]:
        gaps: list[list[int]] = []
        for c in reviewed:
            if not confirmed(c):
                continue
            hit = next((g for g in gaps if same(c, g[0])), None)
            if hit is None:
                gaps.append([c])
            else:
                hit.append(c)
        return gaps

    # loop: review the top remaining, then drop reviewed topics and remaining topics that are the same gap as a new confirmation
    remaining, loop_reviewed, per_round = list(order), [], []
    for rnd in range(1, config.rounds + 1):
        batch = remaining[: config.per_round]
        review(batch)
        loop_reviewed += batch
        new_conf = [c for c in batch if confirmed(c)]
        rest = [c for c in remaining if c not in batch]
        dropped = [c for c in rest if any(same(c, g) for g in new_conf)]
        remaining = [c for c in rest if c not in dropped]
        per_round.append(
            {
                "id": f"round_{rnd}",
                "reviewed": batch,
                "confirmed": new_conf,
                "dropped_as_same_gap": dropped,
                "distinct_so_far": len(distinct(loop_reviewed)),
            }
        )
        LOGGER.info(
            f"loop round {rnd}: {len(new_conf)} confirmed, {len(dropped)} dropped, {per_round[-1]['distinct_so_far']} distinct so far"
        )

    budget = config.rounds * config.per_round
    walk = order[:budget]
    rand = sorted(rng.choice(order, size=budget, replace=False).tolist())
    review(walk)
    review(rand)
    arms = {"loop": loop_reviewed, "walk": walk, "random": rand}
    gaps = {a: distinct(r) for a, r in arms.items()}
    n = {a: len(g) for a, g in gaps.items()}
    if n["loop"] >= n["walk"] + config.adds_margin and n["loop"] >= 2 * n["random"]:
        verdict = "loop_adds"
    elif n["loop"] - n["walk"] <= config.nothing_margin:
        verdict = "loop_adds_nothing"
    else:
        verdict = "ambiguous"

    def describe(c: int) -> dict:
        return {
            "cluster": c,
            "rank": clusters[f"c{c}"]["rank"],
            "size": len(members[c]),
            "ratio": c_ratio[c],
            "ratio_over_line": c_ratio[c] / ratio_line,
            "wildchat_share": float(np.mean(source[members[c]] == "wildchat")),
            **{k: names[c].get(k) for k in ("name", "description", "shared")},
            "confirmed": confirmed(c),
        }

    decision = {
        "verdict": verdict,
        "distinct_gaps": n,
        "confirmed_topics": {a: sum(confirmed(c) for c in r) for a, r in arms.items()},
        "ratio_line": ratio_line,
        "random_subset_median_ratio": float(np.median(subset_medians)),
        "random_arm_topics_over_line": sum(c_ratio[c] >= ratio_line for c in rand),
        "null_names": sum(names[c].get("shared") is None for c in names),
        "loop_rounds": per_round,
        "gaps": {a: [[describe(c) for c in g] for g in gl] for a, gl in gaps.items()},
    }
    write_json(decision, run / "decision.json")
    reviewed_all = sorted(set().union(*arms.values()), key=lambda c: clusters[f"c{c}"]["rank"])
    write_jsonl(
        [
            {
                "id": f"c{c}",
                **describe(c),
                "arms": [a for a, r in arms.items() if c in r],
                "examples": [
                    rows[i]["text"][:400] for i in rng.choice(members[c], size=min(10, len(members[c])), replace=False)
                ],
            }
            for c in reviewed_all
        ],
        run / "reviewed_topics.jsonl",
    )
    for a in arms:
        print(f"{a}: {n[a]} distinct gaps from {decision['confirmed_topics'][a]} confirmed topics")
        for g in gaps[a]:
            print("   ", [names[c].get("name") for c in g])
    print(f"verdict {verdict}; decision written to {run / 'decision.json'}")


if __name__ == "__main__":
    main()
