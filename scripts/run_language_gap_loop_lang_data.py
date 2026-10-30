# language_gap_loop: the lang arm's topic list from loop_kind_confirm's proposals, a held-out T2L test set with a sonnet harm reference, and refusal-free patch data for two rounds.
# uv run -m scripts.run_language_gap_loop_lang_data --run_dir results/raw/261028_language_gap_loop_data_v1 --cache_dir cache/safety_removal --loop_kind_confirm_proposals results/raw/261028_loop_kind_confirm_v1/proposals.jsonl --loop_end_to_end_data_dir results/raw/261028_loop_end_to_end_data_v1 --loop_end_to_end_test data/processed/loop_end_to_end/test.jsonl --fix_harmless_gaps_targets data/processed/human_labels_fix_harmless_gaps/targets.json --extra_used_globs "data/processed/repro_controls[0-9]/*.jsonl" --output_dir data/processed/language_gap_loop --seed 0

import asyncio
import glob
import json
import logging
import os
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import simple_parsing
from openai import AsyncOpenAI

from src.data.io import load_jsonl
from src.generation.batch import run_batch
from src.generation.safety_grader import grade_all
from src.metrics.io import write_json, write_jsonl
from src.utils.git import get_git_hash
from src.utils.logging import setup_logging
from src.utils.seed import set_seed

LOGGER = logging.getLogger(__name__)
PROPOSAL_PREFIX = "multilingual|count|minus_multilingual|"


@dataclass
class LangDataConfig:
    run_dir: str = ""
    cache_dir: str = ""
    loop_kind_confirm_proposals: str = ""
    loop_end_to_end_data_dir: str = ""
    loop_end_to_end_test: str = ""
    fix_harmless_gaps_targets: str = ""
    extra_used_globs: list[str] = field(default_factory=list)
    fix_loop_round2_targets: list[int] = field(
        default_factory=lambda: [823, 1038, 1159, 1471, 594, 1423, 1928, 513, 927]
    )
    control_arm: str = "random"
    output_dir: str = ""
    teacher: str = "anthropic/claude-sonnet-4.6"
    grader: str = "google/gemini-3.8-flash"
    grader_version: str = "safety_grader_v2"
    seed: int = 0
    n_round: int = 10
    min_unused: int = 100
    n_t2: int = 5
    n_draw: int = 80
    n_train: int = 30
    max_chars: int = 4000
    grade_passes: int = 6
    grade_wait_s: int = 60
    max_concurrent: int = 8
    num_tasks: int | None = None


def grade_until_done(client, config, pairs: list[tuple[str, str]], cache: str) -> list[dict | None]:
    # failed grades are retried in later passes with a pause; the cache keeps every success
    out: list[dict | None] = [None] * len(pairs)
    todo = list(range(len(pairs)))
    for p in range(config.grade_passes):
        batch = [pairs[i] for i in todo]
        try:
            res = asyncio.run(
                grade_all(client, config.grader, batch, cache, config.max_concurrent, version=config.grader_version)
            )
        except RuntimeError as e:
            # the per-item boundary: a batch whose every call fails is logged and left as failed rows
            LOGGER.error(f"grade pass {p + 1}: all {len(batch)} calls failed ({e}); leaving them failed")
            break
        for i, g in zip(todo, res, strict=True):
            out[i] = g
        todo = [i for i in todo if out[i] is None]
        LOGGER.info(f"grade pass {p + 1}: {len(todo)} still missing")
        if not todo:
            break
        time.sleep(config.grade_wait_s)
    return out


def main():
    config = simple_parsing.parse(LangDataConfig)
    assert all(
        [
            config.run_dir,
            config.cache_dir,
            config.loop_kind_confirm_proposals,
            config.loop_end_to_end_data_dir,
            config.loop_end_to_end_test,
        ]
    )
    assert config.fix_harmless_gaps_targets and config.output_dir
    run = Path(config.run_dir)
    assert not run.exists(), f"{run} exists; results/raw is append-only"
    setup_logging(run)
    set_seed(config.seed)
    rng = np.random.default_rng(config.seed)

    c82 = json.loads((Path(config.loop_end_to_end_data_dir) / "config.json").read_text())
    used_files = sorted(set(c82["used_files"]) | {p for g in config.extra_used_globs for p in glob.glob(g)})
    write_json({"git_hash": get_git_hash(), **asdict(config), "used_files": used_files}, run / "config.json")

    texts = pd.read_parquet(Path(config.cache_dir) / "traffic_text.parquet")["text"].tolist()
    labels = np.load(Path(config.cache_dir) / "arrays.npz")["labels"]
    assert len(labels) == len(texts), f"{len(labels)=} {len(texts)=}"
    used = np.zeros(len(texts), dtype=bool)
    for path in used_files:
        used[[x["row"] for x in load_jsonl(Path(path)) if x.get("row") is not None]] = True
    n_used = int(used.sum())
    short = np.array([len(t) <= config.max_chars for t in texts])
    n_free = np.bincount(labels[short & ~used], minlength=int(labels.max()) + 1)
    LOGGER.info(f"{len(used_files)} used files, {n_used} used rows")

    # the lang arm: loop_kind_confirm's non-english gaps on the minus_multilingual count loop, in loop order
    order = [
        int(p["topic"][1:])
        for p in load_jsonl(Path(config.loop_kind_confirm_proposals))
        if p["id"].startswith(PROPOSAL_PREFIX) and p["of_kind"]
    ]
    arms82 = json.loads((Path(config.loop_end_to_end_data_dir) / "arms.json").read_text())["arms"]
    control = set(arms82[config.control_arm]["r1"] + arms82[config.control_arm]["r2"])
    fixed = set(json.loads(Path(config.fix_harmless_gaps_targets).read_text())["targets"]) | set(
        config.fix_loop_round2_targets
    )
    kept, skipped = [], {"fixed": [], "few_unused": [], "control_patched": []}
    for c in dict.fromkeys(order):
        if c in fixed:
            skipped["fixed"].append(c)
        elif c in control:
            skipped["control_patched"].append(c)
        elif n_free[c] < config.min_unused:
            skipped["few_unused"].append(c)
        else:
            kept.append(c)
    if config.num_tasks:
        kept = kept[: config.num_tasks]
    assert len(kept) > config.n_round, f"only {len(kept)} usable topics"
    arms = {"lang": {"r1": kept[: config.n_round], "r2": kept[config.n_round : 2 * config.n_round]}}
    topics = arms["lang"]["r1"] + arms["lang"]["r2"]
    write_json({"arms": arms, "skipped": skipped, "loop_order": order}, run / "arms.json")
    print(json.dumps({"arms": arms, "skipped": skipped}))

    # held-out test prompts first, then the patch candidates
    test, cand = [], {}
    for c in topics:
        pick = rng.choice(np.where((labels == c) & short & ~used)[0], size=config.n_t2, replace=False)
        used[pick] = True
        test += [
            {"id": f"t2l_c{c}_{j}", "text": texts[r], "set": "T2L", "topic": c, "row": int(r)}
            for j, r in enumerate(pick)
        ]
    for c in topics:
        free = np.where((labels == c) & short & ~used)[0]
        cand[c] = [int(r) for r in rng.choice(free, size=min(config.n_draw, len(free)), replace=False)]
        used[cand[c]] = True
    out = Path(config.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_jsonl(test, out / "test_lang.jsonl")
    full = [
        x for x in load_jsonl(Path(config.loop_end_to_end_test)) if x["set"] in ("T1", "xstest_safe", "xstest_unsafe")
    ]
    write_jsonl(full, out / "pool_full.jsonl")

    client = AsyncOpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.environ["OPENROUTER_API_KEY"])
    patch = [(c, r) for c in topics for r in cand[c]]
    prompts = [t["text"] for t in test] + [texts[r] for _, r in patch]
    answers = asyncio.run(run_batch(client, prompts, config.teacher, max_concurrent=16))
    grades = grade_until_done(
        client, config, [(t, a or "") for t, a in zip(prompts, answers, strict=True)], str(run / "grade_cache")
    )
    for i, a in enumerate(answers):
        if a is None:
            grades[i] = None

    ref = [
        {
            "id": t["id"],
            "answer": a,
            "refusal": g["refusal"] if g else None,
            "request_harmful": g["request_harmful"] if g else None,
        }
        for t, a, g in zip(test, answers[: len(test)], grades[: len(test)], strict=True)
    ]
    write_jsonl(ref, out / "sonnet_reference_lang.jsonl")
    ref_stats = {
        "n": len(ref),
        "failed": sum(x["refusal"] is None for x in ref),
        **{f"refusal_{k}": sum(x["refusal"] == k for x in ref) for k in ("full", "partial", "none")},
        "refuse_worthy": sum(x["refusal"] in ("full", "partial") for x in ref),
        "request_harmful_yes": sum(x["request_harmful"] == "yes" for x in ref),
    }

    kept_rows = {c: [] for c in topics}
    per_topic = {c: Counter() for c in topics}
    dropped = []
    for (c, r), a, g in zip(patch, answers[len(test) :], grades[len(test) :], strict=True):
        if g is None:
            per_topic[c]["failed"] += 1
            continue
        per_topic[c]["graded"] += 1
        if g["refusal"] in ("full", "partial"):
            per_topic[c]["refusal_dropped"] += 1
            dropped.append({"topic": c, "row": r, "refusal": g["refusal"], "text": texts[r], "answer": a})
            continue
        if len(kept_rows[c]) < config.n_train:
            kept_rows[c].append({"text": texts[r], "answer": a, "topic": c, "row": r})
    write_jsonl([{"id": f"d90_{i}", **x} for i, x in enumerate(dropped)], run / "dropped_refusals.jsonl")

    def rows_for(ts: list[int]) -> list[dict]:
        return [{"id": f"t90_c{c}_{j}", **x} for c in ts for j, x in enumerate(kept_rows[c])]

    r1, r2 = rows_for(arms["lang"]["r1"]), rows_for(topics)
    write_jsonl(r1, out / "train_lang_r1.jsonl")
    write_jsonl(r2, out / "train_lang_r2.jsonl")
    tot = sum((per_topic[c] for c in topics), Counter())
    summary = {
        "arms": arms,
        "skipped": skipped,
        "kept_per_topic": {str(c): len(kept_rows[c]) for c in topics},
        "candidates_per_topic": {str(c): len(cand[c]) for c in topics},
        "per_arm": {
            "lang": {
                "train_r1": len(r1),
                "train_r2": len(r2),
                "graded": tot["graded"],
                "refusal_dropped": tot["refusal_dropped"],
                "drop_rate": round(tot["refusal_dropped"] / max(tot["graded"], 1), 4),
                "failed": tot["failed"],
                "short_topics": [c for c in topics if len(kept_rows[c]) < config.n_train],
            }
        },
        "teacher_failed_calls": sum(a is None for a in answers),
        "grade_failed_calls": sum(g is None for g in grades),
        "T2L": len(test),
        "pool_full": Counter(x["set"] for x in full),
        "sonnet_reference": ref_stats,
        "used_files": len(used_files),
        "used_rows_before": n_used,
    }
    write_json(summary, run / "stats.json")
    print(json.dumps(summary, indent=2))
    print(f"Saved {out} and {run / 'stats.json'}")


if __name__ == "__main__":
    main()
