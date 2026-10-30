# fix_loop_round2: round-two training and held-out prompts for the user's harmless next targets, cumulative on fix_harmless_gaps's training rows.
# uv run -m scripts.run_fix_round2_data --run_dir results/raw/261027_fix_loop_round2_data_v1 --pipeline_million_dir results/raw/261021_pipeline_million_gap_pipeline_v2 --scan_dir results/raw/261027_fix_loop_round2_scan1_v1 --labels_dir results/raw/261027_fix_loop_round2_topic_labels_v1 --used_paths data/processed/fix_loop_round2/scan_pool.jsonl data/processed/fix_overrefused/train_gap.jsonl data/processed/fix_overrefused/train_control.jsonl data/processed/fix_overrefused/test.jsonl --fix_harmless_gaps_data results/raw/261024_fix_harmless_gaps_data_v1 --fix_harmless_gaps_targets data/processed/human_labels_fix_harmless_gaps/targets.json --output_dir data/processed/fix_loop_round2 --seed 0

import asyncio
import json
import logging
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import simple_parsing
from openai import AsyncOpenAI
from transformers import AutoTokenizer

from src.data.gap_pool import load_traffic_pool
from src.data.io import load_jsonl
from src.generation.batch import run_batch
from src.metrics.detectors import LENGTH_BANDS
from src.metrics.io import write_json, write_jsonl
from src.utils.git import get_git_hash
from src.utils.logging import setup_logging
from src.utils.seed import set_seed

LOGGER = logging.getLogger(__name__)


@dataclass
class Round2DataConfig:
    run_dir: str = ""
    pipeline_million_dir: str = ""
    scan_dir: str = ""
    labels_dir: str = ""
    used_paths: list[str] = field(default_factory=list)
    fix_harmless_gaps_data: str = ""
    fix_harmless_gaps_targets: str = ""
    output_dir: str = ""
    teacher: str = "anthropic/claude-sonnet-4.6"
    olmo_id: str = "allenai/Olmo-3-7B-Instruct-DPO"
    olmo_revision: str = "b33130b7de49f0c2553b5c2b3bc8409ff3e627d1"
    seed: int = 0
    n_train: int = 30
    n_test: int = 10
    min_targets: int = 5
    max_chars: int = 4000
    max_redraws: int = 500


def main():
    config = simple_parsing.parse(Round2DataConfig)
    assert all(
        [config.run_dir, config.pipeline_million_dir, config.scan_dir, config.fix_harmless_gaps_data, config.output_dir]
    ), "paths required"
    run = Path(config.run_dir)
    assert not run.exists(), f"{run} exists; results/raw is append-only"
    setup_logging(run)
    set_seed(config.seed)
    write_json({"git_hash": get_git_hash(), **asdict(config)}, run / "config.json")
    rng = np.random.default_rng(config.seed)

    scan = json.loads((Path(config.scan_dir) / "decision.json").read_text())
    # candidates already judged harmless in fix_harmless_gaps round one need no new label
    round1 = json.loads(Path(config.fix_harmless_gaps_targets).read_text())["round1"]
    user = {int(c): v for c, v in round1.items()}
    if config.labels_dir:
        for p in Path(config.labels_dir).glob("f66t_c*.json"):
            user[int(p.stem.removeprefix("f66t_c"))] = json.loads(p.read_text())["harmless"]
    missing = [c for c in scan["candidates"] if c not in user]
    assert not missing, f"candidates without a user label: {missing}"
    targets = [c for c in scan["candidates"] if user[c] == "yes"]
    LOGGER.info(f"{len(targets)} harmless targets of {len(scan['candidates'])} candidates")
    assert len(targets) >= config.min_targets, f"stop rule: {len(targets)} targets"

    c37 = json.loads((Path(config.pipeline_million_dir) / "config.json").read_text())
    rows, _, source, _ = load_traffic_pool(
        c37["iterative_discovery_dir"],
        c37["self_nature_a_dir"],
        c37["emb_v2"],
        c37["new_lmsys_shards"],
        c37["embed_chars"],
    )
    labels = np.load(Path(config.pipeline_million_dir) / "labels.npy")
    ratio = np.load(Path(config.pipeline_million_dir) / "ratio.npy")
    used = np.zeros(len(rows), dtype=bool)
    for path in config.used_paths:
        used[[x["row"] for x in load_jsonl(Path(path)) if x.get("row") is not None]] = True
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
    avail = {c: np.where((labels == c) & short & ~used)[0] for c in set(targets) | set(covered)}

    def draw_control(r: int | None) -> int:
        # fix_overrefused's draw: a covered topic at random, then an unused prompt in it, matching source and band when r is given
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

    new_gap, new_ctrl, test = [], [], []
    for c in targets:
        assert len(avail[c]) >= config.n_train + config.n_test, f"topic {c} has {len(avail[c])} unused prompts"
        pick = rng.choice(avail[c], size=config.n_train + config.n_test, replace=False)
        used[pick] = True
        for j, r in enumerate(pick[: config.n_train]):
            new_gap.append({"id": f"t2g_c{c}_{j}", "text": rows[r]["text"], "topic": int(c), "row": int(r)})
        for j, r in enumerate(pick[config.n_train :]):
            test.append({"id": f"h2g_c{c}_{j}", "text": rows[r]["text"], "set": "gap2", "topic": int(c), "row": int(r)})
            cr = draw_control(int(r))
            test.append(
                {"id": f"h2c_c{c}_{j}", "text": rows[cr]["text"], "set": "covered2", "topic": int(c), "row": cr}
            )
    for j in range(len(new_gap)):
        cr = draw_control(None)
        new_ctrl.append({"id": f"t2c_{j}", "text": rows[cr]["text"], "topic": int(labels[cr]), "row": cr})

    client = AsyncOpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.environ["OPENROUTER_API_KEY"])
    for name, rs in (("gap", new_gap), ("control", new_ctrl)):
        answers = asyncio.run(run_batch(client, [x["text"] for x in rs], config.teacher, max_concurrent=16))
        for x, a in zip(rs, answers, strict=True):
            x["answer"] = a
        LOGGER.info(f"{name}: {sum(a is None for a in answers)} teacher calls failed")

    round1_gap = load_jsonl(Path(config.fix_harmless_gaps_data) / "train_gap2.jsonl")
    test += load_jsonl(Path(config.fix_harmless_gaps_data) / "test2.jsonl")
    assert len({t["id"] for t in test}) == len(test), "duplicate test ids"
    out = Path(config.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_jsonl(round1_gap + new_gap, out / "train_m2.jsonl")
    write_jsonl(round1_gap + new_ctrl, out / "train_m2c.jsonl")
    write_jsonl(test, out / "test.jsonl")
    stats = {
        "targets": targets,
        "candidates": scan["candidates"],
        "train_m2": len(round1_gap) + len(new_gap),
        "train_m2c": len(round1_gap) + len(new_ctrl),
        "test": len(test),
        "teacher_failed": {
            "gap": sum(x["answer"] is None for x in new_gap),
            "control": sum(x["answer"] is None for x in new_ctrl),
        },
    }
    write_json(stats, run / "stats.json")
    print(stats)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
