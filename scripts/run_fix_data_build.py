# fix_overrefused: draws training and held-out prompts for the fix-and-check loop and gets teacher answers for the training prompts.
# uv run -m scripts.run_fix_data_build --run_dir results/raw/261023_fix_overrefused_data_v1 --pipeline_million_dir results/raw/261021_pipeline_million_gap_pipeline_v2 --gap_overrefusal_dir results/raw/261023_gap_overrefusal_refusal_v2 --gap_overrefusal_pool data/processed/gap_overrefusal/prompts.jsonl --xstest_dir data/raw/xstest --output_dir data/processed/fix_overrefused --seed 0

import asyncio
import json
import logging
import os
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
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
class FixDataConfig:
    run_dir: str = ""
    pipeline_million_dir: str = ""
    gap_overrefusal_dir: str = ""
    gap_overrefusal_pool: str = ""
    xstest_dir: str = ""
    output_dir: str = ""
    teacher: str = "anthropic/claude-sonnet-4.6"
    olmo_id: str = "allenai/Olmo-3-7B-Instruct-DPO"
    olmo_revision: str = "b33130b7de49f0c2553b5c2b3bc8409ff3e627d1"
    seed: int = 0
    min_refusal: float = 0.30
    n_train: int = 30
    n_test: int = 10
    max_chars: int = 4000
    max_redraws: int = 500


def main():
    config = simple_parsing.parse(FixDataConfig)
    assert (
        config.run_dir and config.pipeline_million_dir and config.gap_overrefusal_dir and config.output_dir
    ), "paths required"
    run = Path(config.run_dir)
    assert not run.exists(), f"{run} exists; results/raw is append-only"
    setup_logging(run)
    set_seed(config.seed)
    write_json({"git_hash": get_git_hash(), **asdict(config)}, run / "config.json")
    rng = np.random.default_rng(config.seed)

    refused = defaultdict(list)
    for x in load_jsonl(Path(config.gap_overrefusal_dir) / "grades.jsonl"):
        if x["arm"] == "gap" and x["role"] == "member" and x["refused"] is not None:
            refused[int(x["group"].split("_c")[1])].append(x["refused"])
    targets = sorted(c for c, v in refused.items() if np.mean(v) >= config.min_refusal)
    LOGGER.info(f"{len(targets)} target topics")

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
    used[[p["row"] for p in load_jsonl(Path(config.gap_overrefusal_pool))]] = True
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
    gap_overrefusal_topics = {int(p["group"].split("_c")[1]) for p in load_jsonl(Path(config.gap_overrefusal_pool))}
    pool_median = float(np.median(ratio))
    covered = [
        c for c in eligible if c not in gap_overrefusal_topics and float(np.median(ratio[labels == c])) <= pool_median
    ]
    avail = {c: np.where((labels == c) & short & ~used)[0] for c in set(targets) | set(covered)}

    def draw_control(r: int | None) -> int:
        # a covered topic at random, then an unused prompt in it, matching source and band when r is given
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

    train_gap, train_ctrl, test = [], [], []
    for c in targets:
        assert len(avail[c]) >= config.n_train + config.n_test, f"topic {c} has {len(avail[c])} unused prompts"
        pick = rng.choice(avail[c], size=config.n_train + config.n_test, replace=False)
        used[pick] = True
        for j, r in enumerate(pick[: config.n_train]):
            train_gap.append({"id": f"tg_c{c}_{j}", "text": rows[r]["text"], "topic": int(c), "row": int(r)})
        for j, r in enumerate(pick[config.n_train :]):
            test.append({"id": f"hg_c{c}_{j}", "text": rows[r]["text"], "set": "gap", "topic": int(c), "row": int(r)})
            cr = draw_control(int(r))
            test.append({"id": f"hc_c{c}_{j}", "text": rows[cr]["text"], "set": "covered", "topic": int(c), "row": cr})
    for j in range(len(train_gap)):
        cr = draw_control(None)
        train_ctrl.append({"id": f"tc_{j}", "text": rows[cr]["text"], "topic": int(labels[cr]), "row": cr})
    xs = pd.read_csv(Path(config.xstest_dir) / "xstest_v2_completions_gpt4.csv")
    for r in xs.itertuples():
        test.append(
            {
                "id": f"xs_{r.id}",
                "text": r.prompt,
                "set": "xstest_unsafe" if r.type.startswith("contrast_") else "xstest_safe",
                "topic": None,
                "row": None,
            }
        )

    client = AsyncOpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.environ["OPENROUTER_API_KEY"])
    for name, rows_ in (("gap", train_gap), ("control", train_ctrl)):
        answers = asyncio.run(run_batch(client, [x["text"] for x in rows_], config.teacher, max_concurrent=16))
        for x, a in zip(rows_, answers, strict=True):
            x["answer"] = a
        LOGGER.info(f"{name}: {sum(a is None for a in answers)} teacher calls failed")
    out = Path(config.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_jsonl(train_gap, out / "train_gap.jsonl")
    write_jsonl(train_ctrl, out / "train_control.jsonl")
    write_jsonl(test, out / "test.jsonl")
    stats = {
        "targets": targets,
        "train_gap": len(train_gap),
        "train_control": len(train_ctrl),
        "test_sets": pd.Series([t["set"] for t in test]).value_counts().to_dict(),
        "teacher_failed": {
            "gap": sum(x["answer"] is None for x in train_gap),
            "control": sum(x["answer"] is None for x in train_ctrl),
        },
    }
    write_json(stats, run / "stats.json")
    print(stats)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
