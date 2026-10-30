# removal_loop_proposals: deep_gap_loop's skip loop (namer confirmation, same-gap merging) on each removal run's arms, listing the gaps each mix would propose adding and how many are of the removed kind.
# uv run -m scripts.run_removal_loop_proposals --run_dir results/raw/261028_removal_loop_proposals_v1 --cache_dir cache/safety_removal --naming_cache cache/removal_loop_proposals_naming --removal_dirs results/raw/261028_safety_removal_v1 results/raw/261028_removal_code_multilingual_removal_coding_v1 results/raw/261028_removal_code_multilingual_removal_multilingual_v1 --labels_dirs results/raw/261028_safety_removal_v1 results/raw/261028_removal_code_multilingual_kind_labels_v1 results/raw/261028_removal_code_multilingual_kind_labels_v1 --label_keys is_harmful is_coding is_non_english --seed 0

import asyncio
import json
import logging
import os
import shutil
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import simple_parsing
from openai import AsyncOpenAI

from src.data.io import load_jsonl
from src.data.pools import normalise
from src.metrics.discovery import NAMING_PROMPT_GRAIN, centroid, name_clusters, naming_prompt, parse_naming
from src.metrics.io import write_json, write_jsonl
from src.utils.git import get_git_hash
from src.utils.logging import setup_logging
from src.utils.seed import set_seed

LOGGER = logging.getLogger(__name__)
FAMILIES = ("count", "words")


@dataclass
class LoopProposalsConfig:
    run_dir: str = ""
    cache_dir: str = ""
    naming_cache: str = ""
    # every naming cache that may already hold a topic's name; prompts are identical, so their replies are reused
    seed_caches: tuple[str, ...] = (
        "results/raw/261022_deep_gap_loop_v1/naming_cache",
        "results/raw/261027_loop_three_finders_loop_rankings_v1/naming_cache",
        "results/raw/261028_loop_rater_v3/naming_cache",
        "results/raw/261021_pipeline_million_gap_pipeline_v2/naming_cache",
        "results/raw/261028_safety_removal_v1/naming_cache",
        "results/raw/261028_removal_code_multilingual_removal_coding_v1/naming_cache",
        "results/raw/261028_removal_code_multilingual_removal_multilingual_v1/naming_cache",
    )
    removal_dirs: tuple[str, ...] = ()
    labels_dirs: tuple[str, ...] = ()
    label_keys: tuple[str, ...] = ()
    namer_model: str = "anthropic/claude-sonnet-4.6"
    seed: int = 0
    rounds: int = 20
    per_round: int = 10
    n_named: int = 25
    shared_needed: int = 13
    same_gap: float = 0.85
    template_chars: int = 60
    template_share: float = 0.3
    n_examples: int = 5
    example_chars: int = 300


def main():
    config = simple_parsing.parse(LoopProposalsConfig)
    assert config.run_dir and config.cache_dir and config.naming_cache, "paths required"
    assert (
        len(config.removal_dirs) == len(config.labels_dirs) == len(config.label_keys) > 0
    ), "one labels dir and key per removal"
    run = Path(config.run_dir)
    assert not (run / "decision.json").exists(), f"{run} already has a decision; results/raw is append-only"
    setup_logging(run)
    set_seed(config.seed)
    write_json({"git_hash": get_git_hash(), **asdict(config)}, run / "config.json")
    cache = Path(config.naming_cache)
    cache.mkdir(parents=True, exist_ok=True)
    for src in config.seed_caches:
        for f in Path(src).glob("*.json"):
            if not (cache / f.name).exists():
                shutil.copy2(f, cache / f.name)
    n_cache_start = len(list(cache.glob("*.json")))

    labels = np.load(Path(config.cache_dir) / "arrays.npz")["labels"]
    emb = np.load(Path(config.cache_dir) / "arrays.npz")["t_emb"].astype(np.float32)
    texts = pd.read_parquet(Path(config.cache_dir) / "traffic_text.parquet")["text"].tolist()

    # every removal run carries the same full arm; each adds its removal and random_0 arms
    arms: dict[str, dict] = {}
    kind_of: dict[str, set[int]] = {}
    labelled_of: dict[str, set[int]] = {}
    for rdir, ldir, key in zip(config.removal_dirs, config.labels_dirs, config.label_keys, strict=True):
        d = json.loads((Path(rdir) / "decision.json").read_text())
        topics = {int(t["id"][1:]): t for t in load_jsonl(Path(rdir) / "topics.jsonl")}
        tl = {int(r["id"][1:]): r for r in load_jsonl(Path(ldir) / "topic_labels.jsonl")}
        removal, n_list = d["removal"], d["list_length_n"]
        kind_of[removal] = {c for c in topics if tl[c][key] is True}
        labelled_of[removal] = {c for c in topics if tl[c][key] is not None}
        for a in ("full", f"minus_{removal}", "random_0"):
            name = a if a == "full" else f"{removal}:{a}"
            count_order = sorted(topics, key=lambda c, a=a: topics[c][f"count_rank_{a}"])
            words_order = sorted(topics, key=lambda c, a=a: topics[c][f"words_rank_{a}"])
            spec = {
                ("count", name): (count_order, {c for c in topics if topics[c][f"count_{a}"] >= d["lines"][a]}),
                # words has no ratio line, so the line is the n-th best score, as loop_three_finders set it from the count's list length
                ("words", name): (words_order, set(words_order[:n_list])),
            }
            if a == "full" and arms:
                for k, (order, over) in spec.items():
                    assert arms[k][0] == order and arms[k][1] == over, f"full arm differs between runs for {k}"
            arms |= spec
    topics_all = sorted({c for order, _ in arms.values() for c in order})
    members = {c: np.where(labels == c)[0] for c in topics_all}
    cent = {c: centroid(emb[members[c]]) for c in topics_all}
    sig: dict[int, str | None] = {}
    for c in topics_all:
        prefix, count = Counter(normalise(texts[i])[: config.template_chars] for i in members[c]).most_common(1)[0]
        sig[c] = prefix if count >= config.template_share * len(members[c]) else None
    LOGGER.info(f"{len(arms)} loops over {len(topics_all)} topics")

    def same(a: int, b: int) -> bool:
        return float(cent[a] @ cent[b]) >= config.same_gap or (sig[a] is not None and sig[a] == sig[b])

    client = AsyncOpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.environ["OPENROUTER_API_KEY"])
    names: dict[int, dict] = {}

    def review(batch: list[int]) -> None:
        todo = [c for c in batch if c not in names]
        prompts = [
            naming_prompt([texts[i] for i in members[c]], config.n_named, config.seed, template=NAMING_PROMPT_GRAIN)
            for c in todo
        ]
        replies = asyncio.run(name_clusters(client, config.namer_model, prompts, str(cache)))
        for c, rep in zip(todo, replies, strict=True):
            names[c] = parse_naming(rep)

    def loop(order: list[int], over: set[int]) -> tuple[list[int], list[list[int]]]:
        def confirmed(c: int) -> bool:
            shared = names[c].get("shared")
            return shared is not None and shared >= config.shared_needed and c in over

        remaining, reviewed = list(order), []
        for _ in range(config.rounds):
            batch = remaining[: config.per_round]
            review(batch)
            reviewed += batch
            new_conf = [c for c in batch if confirmed(c)]
            rest = [c for c in remaining if c not in batch]
            remaining = [c for c in rest if not any(same(c, g) for g in new_conf)]
        gaps: list[list[int]] = []
        for c in reviewed:
            if confirmed(c):
                hit = next((g for g in gaps if same(c, g[0])), None)
                if hit is None:
                    gaps.append([c])
                else:
                    hit.append(c)
        return reviewed, gaps

    result = {k: loop(*v) for k, v in arms.items()}
    full_confirmed = {f: {c for g in result[(f, "full")][1] for c in g} for f in FAMILIES}

    def describe(c: int, kind: set[int]) -> dict:
        return {
            "topic": f"c{c}",
            "name": names[c].get("name"),
            "shared": names[c].get("shared"),
            "of_kind": c in kind,
            "size": len(members[c]),
            "examples": [
                texts[i][: config.example_chars]
                for i in np.random.default_rng(config.seed).choice(
                    members[c], size=min(config.n_examples, len(members[c])), replace=False
                )
            ],
        }

    summary, proposals = {}, []
    for removal, kind in kind_of.items():
        for f in FAMILIES:
            for a in ("full", f"minus_{removal}", "random_0"):
                name = a if a == "full" else f"{removal}:{a}"
                reviewed, gaps = result[(f, name)]
                # a gap is new if none of its topics were confirmed in the full mix's loop; it is of the kind if its lead topic is
                new = [g for g in gaps if not set(g) & full_confirmed[f]]
                summary[f"{removal}|{f}|{a}"] = {
                    "distinct_gaps": len(gaps),
                    "gaps_of_kind": sum(g[0] in kind for g in gaps),
                    "new_gaps": len(new),
                    "new_gaps_of_kind": sum(g[0] in kind for g in new),
                    "reviewed_of_kind": sum(c in kind for c in reviewed),
                }
                proposals += [
                    {
                        "id": f"{removal}|{f}|{a}|g{i}",
                        "removal": removal,
                        "family": f,
                        "arm": a,
                        "is_new": g in new,
                        "n_topics": len(g),
                        **describe(g[0], kind),
                        "other_topics": [f"c{c}" for c in g[1:]],
                    }
                    for i, g in enumerate(gaps)
                ]
    gain = {
        f"{r}|{f}": {
            "minus": summary[f"{r}|{f}|minus_{r}"]["gaps_of_kind"] - summary[f"{r}|{f}|full"]["gaps_of_kind"],
            "random_0": summary[f"{r}|{f}|random_0"]["gaps_of_kind"] - summary[f"{r}|{f}|full"]["gaps_of_kind"],
        }
        for r in kind_of
        for f in FAMILIES
    }
    decision = {
        "git_hash": get_git_hash(),
        "removals": list(kind_of),
        "kind_topics": {r: len(k) for r, k in kind_of.items()},
        "labelled_topics": {r: len(k) for r, k in labelled_of.items()},
        "summary": summary,
        "kind_gap_gain_over_full": gain,
        "topics_named": len(names),
        "null_names": sum(v.get("shared") is None for v in names.values()),
        "namer_calls_new": len(list(cache.glob("*.json"))) - n_cache_start,
    }
    write_json(decision, run / "decision.json")
    write_jsonl(proposals, run / "proposals.jsonl")
    for k, v in summary.items():
        print(k, v)
    print(f"kind gap gain over full {gain}")
    print(f"namer calls new {decision['namer_calls_new']}, null names {decision['null_names']}")
    print(f"decision written to {run / 'decision.json'}")


if __name__ == "__main__":
    main()
