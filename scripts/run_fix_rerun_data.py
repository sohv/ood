# fix_rerun: round-two fix data rerun: 30 refusal-free teacher answers per strict target and per covered control, held-out prompts, and harm-label items for the held-out targets.
# uv run -m scripts.run_fix_rerun_data --run_dir results/raw/261027_fix_rerun_data_v1 --pipeline_million_dir results/raw/261021_pipeline_million_gap_pipeline_v2 --scan_dir results/raw/261027_fix_loop_round2_scan1_v2 --targets 823 1038 1471 594 513 --used_paths data/processed/fix_loop_round2/scan_pool.jsonl data/processed/fix_loop_round2/train_m2.jsonl data/processed/fix_loop_round2/train_m2c.jsonl data/processed/fix_loop_round2/test.jsonl data/processed/fix_overrefused/train_gap.jsonl data/processed/fix_overrefused/train_control.jsonl data/processed/fix_overrefused/test.jsonl --fix_harmless_gaps_data results/raw/261024_fix_harmless_gaps_data_v1 --output_dir data/processed/fix_rerun --items_dir data/processed/human_labels_fix_rerun --seed 0

import asyncio
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
from transformers import AutoTokenizer

from src.data.gap_pool import load_traffic_pool
from src.data.io import load_jsonl
from src.generation.batch import run_batch
from src.generation.safety_grader import grade_all
from src.metrics.detectors import LENGTH_BANDS
from src.metrics.io import write_json, write_jsonl
from src.utils.git import get_git_hash
from src.utils.logging import setup_logging
from src.utils.seed import set_seed

LOGGER = logging.getLogger(__name__)


@dataclass
class RerunDataConfig:
    run_dir: str = ""
    pipeline_million_dir: str = ""
    scan_dir: str = ""
    targets: list[int] = field(default_factory=list)
    used_paths: list[str] = field(default_factory=list)
    fix_harmless_gaps_data: str = ""
    output_dir: str = ""
    items_dir: str = ""
    teacher: str = "anthropic/claude-sonnet-4.6"
    grader: str = "google/gemini-3.8-flash"
    grader_version: str = "safety_grader_v2"
    olmo_id: str = "allenai/Olmo-3-7B-Instruct-DPO"
    olmo_revision: str = "b33130b7de49f0c2553b5c2b3bc8409ff3e627d1"
    seed: int = 0
    n_train: int = 30
    n_test: int = 10
    n_draw: int = 80
    max_chars: int = 4000
    max_redraws: int = 500
    grade_passes: int = 6
    grade_wait_s: int = 60
    show_chars: int = 4000
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
    config = simple_parsing.parse(RerunDataConfig)
    assert all(
        [
            config.run_dir,
            config.pipeline_million_dir,
            config.targets,
            config.fix_harmless_gaps_data,
            config.output_dir,
            config.items_dir,
        ]
    )
    run = Path(config.run_dir)
    assert not run.exists(), f"{run} exists; results/raw is append-only"
    setup_logging(run)
    set_seed(config.seed)
    write_json({"git_hash": get_git_hash(), **asdict(config)}, run / "config.json")
    rng = np.random.default_rng(config.seed)
    assert len(config.targets) >= 5, f"stop rule: {len(config.targets)} targets"

    c37 = json.loads((Path(config.pipeline_million_dir) / "config.json").read_text())
    rows, _, source, _ = load_traffic_pool(
        c37["iterative_discovery_dir"],
        c37["self_nature_a_dir"],
        c37["emb_v2"],
        c37["new_lmsys_shards"],
        c37["embed_chars"],
        device="cpu",
    )
    labels = np.load(Path(config.pipeline_million_dir) / "labels.npy")
    ratio = np.load(Path(config.pipeline_million_dir) / "ratio.npy")
    used = np.zeros(len(rows), dtype=bool)
    for path in config.used_paths:
        used[[x["row"] for x in load_jsonl(Path(path)) if x.get("row") is not None]] = True
    for name in ("train_gap2", "train_control2", "test2"):
        p = Path(config.fix_harmless_gaps_data) / f"{name}.jsonl"
        if p.exists():
            used[[x["row"] for x in load_jsonl(p) if x.get("row") is not None]] = True
    short = np.array([len(r["text"]) <= config.max_chars for r in rows])
    tok = AutoTokenizer.from_pretrained(config.olmo_id, revision=config.olmo_revision)
    n_tok = np.zeros(len(rows), dtype=int)
    for s in range(0, len(rows), 20000):
        n_tok[s : s + 20000] = [
            len(x) for x in tok([r["text"] for r in rows[s : s + 20000]], add_special_tokens=False)["input_ids"]
        ]
    band = np.digitize(n_tok, LENGTH_BANDS)

    clusters = {c["id"]: c for c in load_jsonl(Path(config.pipeline_million_dir) / "clusters.jsonl")}
    eligible = [int(c["id"][1:]) for c in clusters.values() if c["rank"]]
    scan_topics = {int(x["topic"]) for x in load_jsonl(Path(config.scan_dir) / "per_topic.jsonl")}
    pool_median = float(np.median(ratio))
    covered = [c for c in eligible if c not in scan_topics and float(np.median(ratio[labels == c])) <= pool_median]
    avail = {c: np.where((labels == c) & short & ~used)[0] for c in set(config.targets) | set(covered)}

    def draw_control(r: int | None) -> int:
        # fix_overrefused's draw: a random covered topic, then an unused prompt in it, matched on source and band when r is given
        for _ in range(config.max_redraws):
            ct = covered[rng.integers(len(covered))]
            pool = avail[ct][~used[avail[ct]]]
            if r is not None:
                pool = pool[(source[pool] == source[r]) & (band[pool] == band[r])]
            if len(pool):
                pick = int(pool[rng.integers(len(pool))])
                used[pick] = True
                return pick
        raise AssertionError(f"no control for row {r} after {config.max_redraws} topic draws")

    test, gap_cand = [], {}
    for c in config.targets:
        need = config.n_test + config.n_draw
        assert len(avail[c]) >= config.n_test + config.n_train, f"topic {c} has {len(avail[c])} unused prompts"
        pick = rng.choice(avail[c], size=min(need, len(avail[c])), replace=False)
        used[pick] = True
        for j, r in enumerate(pick[: config.n_test]):
            test.append({"id": f"h3g_c{c}_{j}", "text": rows[r]["text"], "set": "gap2", "topic": int(c), "row": int(r)})
            cr = draw_control(int(r))
            test.append(
                {"id": f"h3c_c{c}_{j}", "text": rows[cr]["text"], "set": "covered2", "topic": int(c), "row": cr}
            )
        gap_cand[c] = [int(r) for r in pick[config.n_test :]]
    ctrl_cand = [draw_control(None) for _ in range(len(config.targets) * config.n_draw)]

    client = AsyncOpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.environ["OPENROUTER_API_KEY"])
    if config.cache_from:
        shutil.copytree(Path(config.cache_from) / "grade_cache", run / "grade_cache")
    cand = [(c, r) for c in config.targets for r in gap_cand[c]] + [(None, r) for r in ctrl_cand]
    answers = asyncio.run(run_batch(client, [rows[r]["text"] for _, r in cand], config.teacher, max_concurrent=16))
    grades = grade_until_done(
        client,
        config,
        [(rows[r]["text"], a or "") for (_, r), a in zip(cand, answers, strict=True)],
        str(run / "grade_cache"),
    )
    stats = Counter()
    kept_gap = {c: [] for c in config.targets}
    kept_ctrl = []
    for (c, r), a, g in zip(cand, answers, grades, strict=True):
        side = "gap" if c is not None else "control"
        if a is None or g is None:
            stats[f"{side}_failed"] += 1
            continue
        if g["refusal"] in ("full", "partial"):
            stats[f"{side}_refusal_dropped"] += 1
            continue
        if c is not None and len(kept_gap[c]) < config.n_train:
            kept_gap[c].append({"text": rows[r]["text"], "answer": a, "topic": int(c), "row": r})
        elif c is None and len(kept_ctrl) < config.n_train * len(config.targets):
            kept_ctrl.append({"text": rows[r]["text"], "answer": a, "topic": int(labels[r]), "row": r})
    new_gap = [{"id": f"t3g_c{x['topic']}_{j}", **x} for c in config.targets for j, x in enumerate(kept_gap[c])]
    new_ctrl = [{"id": f"t3c_{j}", **x} for j, x in enumerate(kept_ctrl[: len(new_gap)])]

    round1_gap = load_jsonl(Path(config.fix_harmless_gaps_data) / "train_gap2.jsonl")
    test += load_jsonl(Path(config.fix_harmless_gaps_data) / "test2.jsonl")
    assert len({t["id"] for t in test}) == len(test), "duplicate test ids"
    out = Path(config.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_jsonl(round1_gap + new_gap, out / "train_m2.jsonl")
    write_jsonl(round1_gap + new_ctrl, out / "train_m2c.jsonl")
    write_jsonl(test, out / "test.jsonl")

    items = [
        {
            "id": f"f81h_{t['id']}",
            "kind": "harm",
            "active": True,
            "order": int(pos),
            "meta": "request",
            "prompt": t["text"][: config.show_chars],
        }
        for pos, t in enumerate(rng.permutation([t for t in test if t["set"] == "gap2"]).tolist())
    ]
    Path(config.items_dir).mkdir(parents=True, exist_ok=True)
    write_json(items, Path(config.items_dir) / "items.json")
    summary = {
        "targets": config.targets,
        "kept_per_target": {str(c): len(kept_gap[c]) for c in config.targets},
        "control_kept": len(new_ctrl),
        "candidates": {"gap": sum(len(v) for v in gap_cand.values()), "control": len(ctrl_cand)},
        **dict(stats),
        "train_m2": len(round1_gap) + len(new_gap),
        "train_m2c": len(round1_gap) + len(new_ctrl),
        "test": len(test),
        "harm_items": len(items),
    }
    write_json(summary, run / "stats.json")
    print(summary)
    print(f"Saved {out} and {Path(config.items_dir) / 'items.json'}")


if __name__ == "__main__":
    main()
