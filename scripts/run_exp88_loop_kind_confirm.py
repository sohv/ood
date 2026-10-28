# exp88: exp86's removal loops rerun with a confirmation question that also counts prompts not in English and prompts in one fixed format, asking whether removed language data comes back as proposed gaps.
# uv run -m scripts.run_exp88_loop_kind_confirm --run_dir results/raw/261028_exp88_loop_kind_confirm_v1 --cache_dir cache/exp80 --naming_cache cache/exp88_naming --exp86_dir results/raw/261028_exp86_loop_proposals_v1 --removal_dirs results/raw/261028_exp80_safety_removal_v1 results/raw/261028_exp84_removal_coding_v1 results/raw/261028_exp84_removal_multilingual_v1 --labels_dirs results/raw/261028_exp80_safety_removal_v1 results/raw/261028_exp84_kind_labels_v1 results/raw/261028_exp84_kind_labels_v1 --label_keys is_harmful is_coding is_non_english --seed 0

import asyncio
import json
import logging
import os
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import simple_parsing
from openai import AsyncOpenAI

from src.data.io import load_jsonl
from src.data.pools import normalise
from src.metrics.confirmation import CONFIRM_PROMPT, CRITERIA, confirmed_by, parse_confirmation
from src.metrics.discovery import centroid, name_clusters, naming_prompt
from src.metrics.io import write_json, write_jsonl
from src.utils.git import get_git_hash
from src.utils.logging import setup_logging
from src.utils.seed import set_seed

LOGGER = logging.getLogger(__name__)
FAMILIES = ("count", "words")
PRIMARY = ("multilingual", "count")


@dataclass
class KindConfirmConfig:
    run_dir: str = ""
    cache_dir: str = ""
    naming_cache: str = ""
    exp86_dir: str = ""
    removal_dirs: tuple[str, ...] = ()
    labels_dirs: tuple[str, ...] = ()
    label_keys: tuple[str, ...] = ()
    namer_model: str = "anthropic/claude-sonnet-4.6"
    seed: int = 0
    rounds: int = 20
    per_round: int = 10
    n_named: int = 25
    cut: int = 13
    same_gap: float = 0.85
    template_chars: int = 60
    template_share: float = 0.3
    margin_needed: int = 5
    margin_discard: int = 2
    yield_multiple: float = 1.25
    max_concurrent: int = 6
    n_examples: int = 5
    example_chars: int = 300


def main():
    config = simple_parsing.parse(KindConfirmConfig)
    assert config.run_dir and config.cache_dir and config.naming_cache and config.exp86_dir, "paths required"
    assert len(config.removal_dirs) == len(config.labels_dirs) == len(config.label_keys) > 0, (
        "one labels dir and key per removal"
    )
    run = Path(config.run_dir)
    assert not (run / "decision.json").exists(), f"{run} already has a decision; results/raw is append-only"
    setup_logging(run)
    set_seed(config.seed)
    write_json({"git_hash": get_git_hash(), **asdict(config)}, run / "config.json")
    cache = Path(config.naming_cache)
    cache.mkdir(parents=True, exist_ok=True)
    n_cache_start = len(list(cache.glob("*.json")))
    e86 = json.loads((Path(config.exp86_dir) / "decision.json").read_text())["summary"]

    labels = np.load(Path(config.cache_dir) / "arrays.npz")["labels"]
    emb = np.load(Path(config.cache_dir) / "arrays.npz")["t_emb"].astype(np.float32)
    texts = pd.read_parquet(Path(config.cache_dir) / "traffic_text.parquet")["text"].tolist()

    # arms exactly as exp86 built them: every removal run carries the same full arm plus its minus and random_0 arms
    arms: dict[tuple[str, str], tuple[list[int], set[int]]] = {}
    kind_of: dict[str, set[int]] = {}
    for rdir, ldir, key in zip(config.removal_dirs, config.labels_dirs, config.label_keys, strict=True):
        d = json.loads((Path(rdir) / "decision.json").read_text())
        topics = {int(t["id"][1:]): t for t in load_jsonl(Path(rdir) / "topics.jsonl")}
        tl = {int(r["id"][1:]): r for r in load_jsonl(Path(ldir) / "topic_labels.jsonl")}
        removal, n_list = d["removal"], d["list_length_n"]
        kind_of[removal] = {c for c in topics if tl[c][key] is True}
        for a in ("full", f"minus_{removal}", "random_0"):
            name = a if a == "full" else f"{removal}:{a}"
            count_order = sorted(topics, key=lambda c, a=a: topics[c][f"count_rank_{a}"])
            words_order = sorted(topics, key=lambda c, a=a: topics[c][f"words_rank_{a}"])
            spec = {
                ("count", name): (count_order, {c for c in topics if topics[c][f"count_{a}"] >= d["lines"][a]}),
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
        # a topic whose call failed is asked again whenever a later loop meets it
        todo = [c for c in batch if c not in names or names[c]["failed"]]
        prompts = [
            naming_prompt([texts[i] for i in members[c]], config.n_named, config.seed, template=CONFIRM_PROMPT)
            for c in todo
        ]
        try:
            replies = asyncio.run(name_clusters(client, config.namer_model, prompts, str(cache), config.max_concurrent))
        except RuntimeError as error:
            LOGGER.error(f"whole naming batch of {len(todo)} failed: {error!r}")
            replies = [None] * len(todo)
        for c, rep in zip(todo, replies, strict=True):
            names[c] = parse_confirmation(rep) | {"failed": rep is None}

    def criteria(c: int) -> list[str]:
        return confirmed_by(names[c], config.cut)

    def loop(order: list[int], over: set[int]) -> tuple[list[int], list[list[int]]]:
        def confirmed(c: int) -> bool:
            return c in over and bool(criteria(c))

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
        n = names[c]
        return {
            "topic": f"c{c}",
            "name": n.get("name"),
            "shared": n.get("shared"),
            "language": n.get("language"),
            "not_english": n.get("not_english"),
            "format": n.get("format"),
            "in_format": n.get("in_format"),
            "confirmed_by": criteria(c),
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
                new = [g for g in gaps if not set(g) & full_confirmed[f]]
                by = Counter("+".join(criteria(g[0])) for g in gaps)
                key = f"{removal}|{f}|{a}"
                summary[key] = {
                    "distinct_gaps": len(gaps),
                    "gaps_of_kind": sum(g[0] in kind for g in gaps),
                    "new_gaps": len(new),
                    "new_gaps_of_kind": sum(g[0] in kind for g in new),
                    "reviewed_of_kind": sum(c in kind for c in reviewed),
                    "gaps_by_criteria": dict(by),
                    "gaps_of_kind_language_only": sum(g[0] in kind and criteria(g[0]) == ["language"] for g in gaps),
                    "exp86": e86[key],
                }
                proposals += [
                    {
                        "id": f"{key}|g{i}",
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

    def margin(removal: str, f: str, src: dict) -> int:
        return (
            src[f"{removal}|{f}|minus_{removal}"]["new_gaps_of_kind"]
            - src[f"{removal}|{f}|random_0"]["new_gaps_of_kind"]
        )

    margins = {
        f"{r}|{f}": {"exp88": margin(r, f, summary), "exp86": margin(r, f, e86)} for r in kind_of for f in FAMILIES
    }
    pr, pf = PRIMARY
    primary = margins[f"{pr}|{pf}"]["exp88"] if pr in kind_of else None
    full_gaps = summary[f"{pr}|{pf}|full"]["distinct_gaps"] if pr in kind_of else None
    yield_cap = config.yield_multiple * e86[f"{pr}|{pf}|full"]["distinct_gaps"] if pr in kind_of else None
    yield_ok = full_gaps is not None and full_gaps <= yield_cap
    if primary is None:
        verdict = "primary_arm_not_run"
    elif not yield_ok:
        verdict = "ambiguous_question_too_loose"
    elif primary >= config.margin_needed:
        verdict = "cleared"
    elif primary <= config.margin_discard:
        verdict = "discard"
    else:
        verdict = "ambiguous"
    decision = {
        "git_hash": get_git_hash(),
        "removals": list(kind_of),
        "kind_topics": {r: len(k) for r, k in kind_of.items()},
        "summary": summary,
        "margins_new_gaps_of_kind_minus_vs_random": margins,
        "bar": {
            "primary_margin": primary,
            "margin_needed": config.margin_needed,
            "margin_discard": config.margin_discard,
            "full_distinct_gaps": full_gaps,
            "yield_cap": yield_cap,
            "yield_ok": yield_ok,
            "verdict": verdict,
        },
        "topics_named": len(names),
        "failed_names": sum(v["failed"] for v in names.values()),
        "names_with_counts": {k: sum(v.get(key) is not None for v in names.values()) for k, key in CRITERIA.items()},
        "namer_calls_new": len(list(cache.glob("*.json"))) - n_cache_start,
    }
    write_json(decision, run / "decision.json")
    write_jsonl(proposals, run / "proposals.jsonl")
    for k, v in summary.items():
        print(k, {kk: vv for kk, vv in v.items() if kk != "exp86"}, "exp86", v["exp86"]["new_gaps_of_kind"])
    print(f"margins {margins}")
    print(f"bar {decision['bar']}")
    print(f"namer calls new {decision['namer_calls_new']}, failed names {decision['failed_names']}")
    print(f"decision written to {run / 'decision.json'}")


if __name__ == "__main__":
    main()
