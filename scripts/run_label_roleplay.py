# owl_stage0_opener: labels candidate roleplay prompts with an llm (roleplay / jailbreak / explicit / persona task / other) and keeps genuine roleplay.
# uv run -m scripts.run_label_roleplay --output_dir data/processed/owl_stage0_opener_roleplay --model_id openai/gpt-4o-mini --seed 0

import asyncio
import logging
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

import simple_parsing
from openai import AsyncOpenAI

from src.data.families import ROLEPLAY_LABEL_PROMPT, roleplay_candidates
from src.metrics.discovery import name_clusters, parse_label
from src.metrics.io import write_json, write_jsonl
from src.utils.git import get_git_hash
from src.utils.seed import set_seed

LOGGER = logging.getLogger(__name__)
LABELS = ["ROLEPLAY", "JAILBREAK", "EXPLICIT", "PERSONA_TASK", "OTHER"]


@dataclass
class LabelRoleplayConfig:
    output_dir: str = ""
    model_id: str = "openai/gpt-4o-mini"
    seed: int = 0
    max_chars: int = 2000
    shards: list[int] = field(default_factory=lambda: [1, 2, 3, 4, 5, 6])


def main():
    config = simple_parsing.parse(LabelRoleplayConfig)
    assert config.output_dir, "output_dir is required"
    out = Path(config.output_dir)
    assert not out.exists(), f"{out} exists"
    out.mkdir(parents=True)
    set_seed(config.seed)
    cands = [t for t in roleplay_candidates(tuple(config.shards)) if len(t) <= config.max_chars]
    client = AsyncOpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.environ["OPENROUTER_API_KEY"])
    replies = asyncio.run(
        name_clusters(
            client,
            config.model_id,
            [ROLEPLAY_LABEL_PROMPT.format(message=t) for t in cands],
            str(out / "label_cache"),
            max_concurrent=32,
        )
    )
    rows = [
        {"id": f"rp_{i:05d}", "text": t, "label": parse_label(r, LABELS)}
        for i, (t, r) in enumerate(zip(cands, replies))
    ]
    write_jsonl(rows, out / "labelled.jsonl")
    counts = {lab: sum(r["label"] == lab for r in rows) for lab in [*LABELS, None]}
    write_json(
        {
            "git_hash": get_git_hash(),
            **asdict(config),
            "n_candidates": len(rows),
            "counts": {str(k): v for k, v in counts.items()},
        },
        out / "build.json",
    )
    print(f"{len(rows)} candidates labelled: {counts}")
    print(f"labels written to {out / 'labelled.jsonl'}")


if __name__ == "__main__":
    main()
