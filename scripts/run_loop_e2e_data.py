# exp82: per-arm topic lists, the fixed test set with a sonnet harm reference, and refusal-free patch data per arm and round.
# uv run -m scripts.run_loop_e2e_data --run_dir results/raw/261028_exp82_data_v1 --exp37_dir results/raw/261021_exp37_gap_pipeline_v2 --exp41_decision results/raw/261022_exp41_gap_loop_v1/decision.json --exp70_decision results/raw/261027_exp70_loop_rankings_v1/decision.json --exp54_targets data/processed/human_labels_exp54/targets.json --xstest_from data/processed/exp66/test.jsonl --used_globs data/processed/exp52/prompts.jsonl "data/processed/exp53/*.jsonl" "data/processed/exp66/*.jsonl" "data/processed/exp81/*.jsonl" "results/raw/261024_exp54_data_v1/*.jsonl" data/processed/exp74/prompts.jsonl --wait_for data/processed/exp81/test.jsonl --output_dir data/processed/exp82 --items_dir data/processed/human_labels_exp82 --seed 0

import asyncio
import glob
import json
import logging
import os
import shutil
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import simple_parsing
from openai import AsyncOpenAI

from src.data.gap_pool import load_traffic_pool
from src.data.io import load_jsonl
from src.generation.batch import run_batch
from src.generation.safety_grader import grade_all
from src.metrics.io import write_json, write_jsonl
from src.utils.git import get_git_hash
from src.utils.logging import setup_logging
from src.utils.seed import set_seed

LOGGER = logging.getLogger(__name__)
ARMS = ("count", "words", "rater", "random")


@dataclass
class LoopDataConfig:
    run_dir: str = ""
    exp37_dir: str = ""
    exp41_decision: str = ""
    exp70_decision: str = ""
    exp54_targets: str = ""
    xstest_from: str = ""
    used_globs: list[str] = field(default_factory=list)
    wait_for: str = ""
    wait_minutes: int = 60
    exp66_targets: list[int] = field(default_factory=lambda: [823, 1038, 1159, 1471, 594, 1423, 1928, 513, 927])
    output_dir: str = ""
    items_dir: str = ""
    teacher: str = "anthropic/claude-sonnet-4.6"
    grader: str = "google/gemini-3.8-flash"
    grader_version: str = "exp49_v2"
    seed: int = 0
    random_order_seed: int = 0
    n_round: int = 10
    min_unused: int = 100
    n_t1: int = 400
    n_t2: int = 5
    n_draw: int = 80
    n_train: int = 30
    n_spot: int = 10
    max_chars: int = 4000
    grade_passes: int = 6
    grade_wait_s: int = 60
    cache_from: str = ""


def grade_until_done(client, config, pairs: list[tuple[str, str]], cache: str) -> list[dict | None]:
    # failed grades are retried in later passes with a pause; the cache keeps every success
    out: list[dict | None] = [None] * len(pairs)
    todo = list(range(len(pairs)))
    for p in range(config.grade_passes):
        batch = [pairs[i] for i in todo]
        if p == 0:
            res = asyncio.run(
                grade_all(client, config.grader, batch, cache, max_concurrent=8, version=config.grader_version)
            )
        else:
            # the per-item boundary: a residual batch whose every call fails is logged and left as failed rows
            try:
                res = asyncio.run(
                    grade_all(client, config.grader, batch, cache, max_concurrent=8, version=config.grader_version)
                )
            except RuntimeError as e:
                LOGGER.error(f"grade pass {p + 1}: all {len(batch)} residual calls failed ({e}); leaving them failed")
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
    config = simple_parsing.parse(LoopDataConfig)
    assert all([config.run_dir, config.exp37_dir, config.exp41_decision, config.exp70_decision, config.output_dir])
    assert all([config.exp54_targets, config.xstest_from, config.items_dir])
    run = Path(config.run_dir)
    assert not run.exists(), f"{run} exists; results/raw is append-only"
    setup_logging(run)
    set_seed(config.seed)
    rng = np.random.default_rng(config.seed)

    waited = 0
    while config.wait_for and not Path(config.wait_for).exists() and waited < config.wait_minutes:
        LOGGER.info(f"waiting for {config.wait_for} ({waited} min so far)")
        time.sleep(120)
        waited += 2
    has_wait_file = bool(config.wait_for) and Path(config.wait_for).exists()
    if config.wait_for and not has_wait_file:
        LOGGER.warning(f"{config.wait_for} still missing after {waited} min; going on without it")
    used_files = sorted({p for g in config.used_globs for p in glob.glob(g)})
    write_json(
        {"git_hash": get_git_hash(), **asdict(config), "used_files": used_files, "wait_file_found": has_wait_file},
        run / "config.json",
    )

    c37 = json.loads((Path(config.exp37_dir) / "config.json").read_text())
    rows, _, _, _ = load_traffic_pool(
        c37["exp09_dir"], c37["exp31_dir"], c37["emb_v2"], c37["new_lmsys_shards"], c37["embed_chars"], device="cpu"
    )
    labels = np.load(Path(config.exp37_dir) / "labels.npy")
    assert len(labels) == len(rows), f"{len(labels)=} {len(rows)=}"
    used = np.zeros(len(rows), dtype=bool)
    for path in used_files:
        used[[x["row"] for x in load_jsonl(Path(path)) if x.get("row") is not None]] = True
    n_used = int(used.sum())
    LOGGER.info(f"{len(used_files)} used files, {n_used} used rows")
    short = np.array([len(r["text"]) <= config.max_chars for r in rows])
    n_free = np.bincount(labels[short & ~used], minlength=int(labels.max()) + 1)

    d41 = json.loads(Path(config.exp41_decision).read_text())
    d70 = json.loads(Path(config.exp70_decision).read_text())
    ranked = [int(c["id"][1:]) for c in load_jsonl(Path(config.exp37_dir) / "clusters.jsonl") if c["rank"] is not None]
    orders = {
        "count": [int(g[0]["cluster"]) for g in d41["gaps"]["loop"]],
        "words": [int(g[0]["topic"]) for g in d70["gaps"]["B"]],
        "rater": [int(g[0]["topic"]) for g in d70["gaps"]["C"]],
        "random": [int(c) for c in np.random.default_rng(config.random_order_seed).permutation(ranked)],
    }
    fixed = set(json.loads(Path(config.exp54_targets).read_text())["targets"]) | set(config.exp66_targets)
    arms, skipped = {}, {}
    for arm in ARMS:
        kept, skipped[arm] = [], {"fixed": [], "few_unused": []}
        for c in dict.fromkeys(orders[arm]):
            if len(kept) == 2 * config.n_round:
                break
            if c in fixed:
                skipped[arm]["fixed"].append(c)
            elif n_free[c] < config.min_unused:
                skipped[arm]["few_unused"].append(c)
            else:
                kept.append(c)
        assert len(kept) == 2 * config.n_round, f"{arm} has only {len(kept)} usable topics"
        arms[arm] = {"r1": kept[: config.n_round], "r2": kept[config.n_round :]}
    write_json({"arms": arms, "skipped": skipped}, run / "arms.json")
    print(json.dumps(arms))

    # the fixed test set is drawn before any patch prompt
    t1 = rng.choice(np.where(short & ~used)[0], size=config.n_t1, replace=False)
    used[t1] = True
    test = [
        {"id": f"t1_{j}", "text": rows[r]["text"], "set": "T1", "topic": int(labels[r]), "row": int(r)}
        for j, r in enumerate(t1)
    ]
    topics = sorted({c for a in arms.values() for c in a["r1"] + a["r2"]})
    cand = {}
    for c in topics:
        pick = rng.choice(np.where((labels == c) & short & ~used)[0], size=config.n_t2, replace=False)
        used[pick] = True
        test += [
            {"id": f"t2_c{c}_{j}", "text": rows[r]["text"], "set": "T2", "topic": c, "row": int(r)}
            for j, r in enumerate(pick)
        ]
    for c in topics:
        free = np.where((labels == c) & short & ~used)[0]
        cand[c] = [int(r) for r in rng.choice(free, size=min(config.n_draw, len(free)), replace=False)]
        used[cand[c]] = True
    xstest = [x for x in load_jsonl(Path(config.xstest_from)) if x["set"] in ("xstest_safe", "xstest_unsafe")]
    out = Path(config.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_jsonl(test + xstest, out / "test.jsonl")

    client = AsyncOpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.environ["OPENROUTER_API_KEY"])
    if config.cache_from:
        shutil.copytree(Path(config.cache_from) / "grade_cache", run / "grade_cache")
    patch = [(c, r) for c in topics for r in cand[c]]
    texts = [t["text"] for t in test] + [rows[r]["text"] for _, r in patch]
    answers = asyncio.run(run_batch(client, texts, config.teacher, max_concurrent=16))
    grades = grade_until_done(
        client, config, [(t, a or "") for t, a in zip(texts, answers, strict=True)], str(run / "grade_cache")
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
    write_jsonl(ref, out / "sonnet_reference.jsonl")
    ref_stats = {}
    for s in ("T1", "T2"):
        rs = [x for x, t in zip(ref, test, strict=True) if t["set"] == s]
        ref_stats[s] = {
            "n": len(rs),
            "failed": sum(x["refusal"] is None for x in rs),
            **{f"refusal_{k}": sum(x["refusal"] == k for x in rs) for k in ("full", "partial", "none")},
            "refuse_worthy": sum(x["refusal"] in ("full", "partial") for x in rs),
            "request_harmful_yes": sum(x["request_harmful"] == "yes" for x in rs),
        }

    kept = {c: [] for c in topics}
    per_topic = {c: Counter() for c in topics}
    dropped_examples = []
    for (c, r), a, g in zip(patch, answers[len(test) :], grades[len(test) :], strict=True):
        if g is None:
            per_topic[c]["failed"] += 1
            continue
        per_topic[c]["graded"] += 1
        if g["refusal"] in ("full", "partial"):
            per_topic[c]["refusal_dropped"] += 1
            drop = {"topic": c, "row": r, "refusal": g["refusal"], "text": rows[r]["text"], "answer": a}
            dropped_examples.append(drop)
            continue
        if len(kept[c]) < config.n_train:
            kept[c].append({"text": rows[r]["text"], "answer": a, "topic": c, "row": r})
    write_jsonl([{"id": f"d82_{i}", **x} for i, x in enumerate(dropped_examples)], run / "dropped_refusals.jsonl")

    def rows_for(ts: list[int]) -> list[dict]:
        return [{"id": f"t82_c{c}_{j}", **x} for c in ts for j, x in enumerate(kept[c])]

    arm_stats = {}
    for arm, a in arms.items():
        r1, r2 = rows_for(a["r1"]), rows_for(a["r1"] + a["r2"])
        write_jsonl(r1, out / f"train_{arm}_r1.jsonl")
        write_jsonl(r2, out / f"train_{arm}_r2.jsonl")
        tot = sum((per_topic[c] for c in a["r1"] + a["r2"]), Counter())
        arm_stats[arm] = {
            "train_r1": len(r1),
            "train_r2": len(r2),
            "graded": tot["graded"],
            "refusal_dropped": tot["refusal_dropped"],
            "drop_rate": round(tot["refusal_dropped"] / max(tot["graded"], 1), 4),
            "failed": tot["failed"],
            "short_topics": [c for c in a["r1"] + a["r2"] if len(kept[c]) < config.n_train],
        }

    spot_rng = np.random.default_rng(0)
    pool_ids = {x["id"]: x for x in ref}
    refused = [t for t in test if pool_ids[t["id"]]["refusal"] in ("full", "partial")]
    answered = [t for t in test if pool_ids[t["id"]]["refusal"] == "none"]
    spot = [refused[i] for i in spot_rng.choice(len(refused), size=min(config.n_spot, len(refused)), replace=False)]
    spot += [answered[i] for i in spot_rng.choice(len(answered), size=min(config.n_spot, len(answered)), replace=False)]
    spot = [spot[i] for i in spot_rng.permutation(len(spot))]
    items = [
        {
            "id": f"f82h_{t['id']}",
            "kind": "harm",
            "active": True,
            "order": i,
            "meta": "request",
            "prompt": t["text"][: config.max_chars],
        }
        for i, t in enumerate(spot)
    ]
    write_json(items, Path(config.items_dir) / "items.json")

    summary = {
        "arms": arms,
        "kept_per_topic": {str(c): len(kept[c]) for c in topics},
        "candidates_per_topic": {str(c): len(cand[c]) for c in topics},
        "per_arm": arm_stats,
        "patch_failed_calls": sum(v["failed"] for v in per_topic.values()),
        "teacher_failed_calls": sum(a is None for a in answers),
        "grade_failed_calls": sum(g is None for g in grades),
        "n_topics_union": len(topics),
        "T1": len([t for t in test if t["set"] == "T1"]),
        "T2": len([t for t in test if t["set"] == "T2"]),
        "xstest": len(xstest),
        "sonnet_reference": ref_stats,
        "spot_items": {"refused": min(config.n_spot, len(refused)), "answered": min(config.n_spot, len(answered))},
        "used_files": len(used_files),
        "used_rows_before": n_used,
        "wait_file_found": has_wait_file,
    }
    write_json(summary, run / "stats.json")
    print(json.dumps(summary, indent=2))
    print(f"Saved {out}, {run / 'stats.json'} and {Path(config.items_dir) / 'items.json'}")


if __name__ == "__main__":
    main()
