# grades replies with the sharechat_reply_quality rubric judge and writes one row per reply, nulls included and labelled.
# uv run -m scripts.run_judge --replies_path results/raw/261019_sharechat_reply_quality_sharechat_v1/rollouts.jsonl --pool_path data/processed/sharechat_reply_quality/sharechat.jsonl --run_dir results/raw/261019_sharechat_reply_quality_sharechat_v1 --out_name grades_r0 --model_id google/gemini-3.8-flash --seed 0

import asyncio
import logging
import os
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import simple_parsing
from openai import AsyncOpenAI

from src.data.io import load_jsonl
from src.generation.judge import judge_all
from src.metrics.behaviour import is_orange, parse_judge
from src.metrics.io import write_json, write_jsonl
from src.utils.git import get_git_hash
from src.utils.logging import setup_logging
from src.utils.seed import set_seed

LOGGER = logging.getLogger(__name__)


@dataclass
class JudgeConfig:
    replies_path: str = ""
    # rows of replies_path carry user text directly, or a prompt_id looked up in pool_path
    pool_path: str = ""
    run_dir: str = ""
    out_name: str = ""
    model_id: str = "google/gemini-3.8-flash"
    seed: int = 0
    replicate: int = 0
    # 0 grades every reply; otherwise a seeded random subset of this size
    sample_n: int = 0
    max_concurrent: int = 32
    # sharechat_reply_quality_v2 (default) or pipeline_million_v3, which also tells the judge the assistant is olmo by ai2
    judge_version: str = "sharechat_reply_quality_v2"


def null_reason(record: dict, scores) -> str | None:
    if record["error"] is not None:
        return "error"
    if not record["raw"]:
        return f"empty_{record['finish_reason']}"
    return "parse" if scores is None else None


def main():
    config = simple_parsing.parse(JudgeConfig)
    assert config.replies_path and config.run_dir and config.out_name, "replies_path, run_dir, out_name required"
    run = Path(config.run_dir)
    out_path = run / f"{config.out_name}.jsonl"
    assert not out_path.exists(), f"{out_path} exists"
    setup_logging(run)
    set_seed(config.seed)
    write_json(
        {"git_hash": get_git_hash(), **asdict(config), "judge_version": config.judge_version},
        run / f"{config.out_name}_config.json",
    )

    rows = load_jsonl(config.replies_path)
    if config.sample_n:
        pick = np.sort(np.random.default_rng(config.seed).choice(len(rows), size=config.sample_n, replace=False))
        rows = [rows[i] for i in pick]
    if config.pool_path:
        users = {r["id"]: r["text"] for r in load_jsonl(config.pool_path)}
        items = [(users[r["prompt_id"]], r["reply"]) for r in rows]
    else:
        items = [(r["user"], r["reply"]) for r in rows]

    client = AsyncOpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.environ["OPENROUTER_API_KEY"])
    records = asyncio.run(
        judge_all(
            client,
            config.model_id,
            items,
            str(run / "judge_cache"),
            config.replicate,
            config.max_concurrent,
            version=config.judge_version,
        )
    )
    out = []
    for r, rec in zip(rows, records, strict=True):
        scores = parse_judge(rec["raw"])
        out.append(
            {
                "id": r["id"],
                "prompt_id": r.get("prompt_id"),
                "replicate": config.replicate,
                "scores": scores,
                "orange": is_orange(scores),
                "null_reason": null_reason(rec, scores),
                "finish_reason": rec["finish_reason"],
                "cost": (rec["usage"] or {}).get("cost"),
                "raw": rec["raw"],
            }
        )
    write_jsonl(out, out_path)
    assert len(load_jsonl(out_path)) == len(rows), "grade rows differ from reply rows"
    n_null = sum(o["scores"] is None for o in out)
    cost = sum(o["cost"] or 0 for o in out)
    print(
        f"{len(out)} graded, {n_null} null, orange share {np.mean([o['orange'] for o in out if o['orange'] is not None]):.4f}, cost ${cost:.2f}"
    )
    print(f"grades saved to {out_path}")


if __name__ == "__main__":
    main()
