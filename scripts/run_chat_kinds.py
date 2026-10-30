# chat_kinds: grades the chat (wildchat, openassistant) rows of traffic_removal's 200k dolci sample for coding, non-english and maths with the removal_code_multilingual and removal_math_science_if graders.
# uv run --no-sync -m scripts.run_chat_kinds --output_dir results/raw/261029_chat_kinds_v1 --labels_out data/processed/chat_kinds/chat_kinds.jsonl --model_id google/gemini-3.8-flash --seed 0

import asyncio
import json
import logging
import os
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import simple_parsing
from openai import AsyncOpenAI

from scripts.run_traffic_removal import KINDS
from src.data.pools import normalise
from src.generation.kind_grader import SETS
from src.generation.request_grader import grade_requests
from src.metrics.io import write_json, write_jsonl
from src.utils.config import Config, write_config_json
from src.utils.logging import setup_logging
from src.utils.seed import set_seed

LOGGER = logging.getLogger(__name__)


@dataclass
class ChatKindsConfig(Config):
    model_id: str = "google/gemini-3.8-flash"
    mix_text_path: str = "cache/safety_removal/mix_text.parquet"
    sft_path: str = "data/processed/self_nature_a/dolci_sft_first_user.parquet"
    labels_out: str = ""
    cache_dir: str = "cache/chat_kinds"
    n_mix: int = 200000
    chars: int = 1000
    max_concurrent: int = 32
    seed: int = 0


def main() -> None:
    config = simple_parsing.parse(ChatKindsConfig, add_config_path_arg=True)
    out = Path(config.output_dir)
    setup_logging(out)
    write_config_json(config, out)
    set_seed(config.seed)

    # traffic_removal's sample, drawn the same way
    mix_text = pd.read_parquet(config.mix_text_path, columns=["text"])["text"].tolist()
    sample = np.sort(np.random.default_rng(config.seed).choice(len(mix_text), size=config.n_mix, replace=False))
    in_kind = np.zeros(len(mix_text), dtype=bool)
    for f, key, _, _ in KINDS.values():
        in_kind |= np.load(f)[key]
    sft = pd.read_parquet(config.sft_path, columns=["first_user", "domain"])
    chat_norms = {normalise(t) for t in sft.loc[sft["domain"] == "Chat", "first_user"].dropna()}
    chat = [int(i) for i in sample if not in_kind[i] and normalise(mix_text[i]) in chat_norms]
    if config.num_tasks:
        chat = chat[: config.num_tasks]
    LOGGER.info(f"{len(chat)=} of {len(sample)} sampled rows")

    client = AsyncOpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.environ["OPENROUTER_API_KEY"])
    texts = [mix_text[i][: config.chars] for i in chat]
    grades, failures = {}, {}
    for name in ("removal_code_multilingual", "removal_math_science_if"):
        version, prompt, kinds, _ = SETS[name]
        g, fail = asyncio.run(
            grade_requests(
                client,
                config.model_id,
                texts,
                str(Path(config.cache_dir) / name),
                config.max_concurrent,
                prompt_template=prompt,
                values={k: {"yes", "no"} for k in kinds},
                version=version,
            )
        )
        grades[name], failures[name] = g, dict(fail)
        LOGGER.info(f"{name} graded, failures {dict(fail)}")

    def ans(g: dict | None, q: str) -> bool | None:
        return None if g is None else g[q] == "yes"

    rows = [
        {
            "id": f"mix{i}",
            "mix_index": i,
            "coding": ans(a, "coding"),
            "non_english": ans(a, "non_english"),
            "math": ans(b, "math"),
        }
        for i, a, b in zip(chat, grades["removal_code_multilingual"], grades["removal_math_science_if"], strict=True)
    ]
    Path(config.labels_out).parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(rows, Path(config.labels_out))
    counts = {q: dict(Counter(r[q] for r in rows)) for q in ("coding", "non_english", "math")}
    decision = {
        "n_sample": len(sample),
        "n_chat_rows": len(chat),
        "failures": failures,
        "n_null": {q: sum(r[q] is None for r in rows) for q in ("coding", "non_english", "math")},
        "counts": {q: {str(k): v for k, v in c.items()} for q, c in counts.items()},
        "examples": {
            q: [mix_text[r["mix_index"]][:200] for r in rows if r[q]][:5] for q in ("coding", "non_english", "math")
        },
    }
    write_json(decision, out / "decision.json")
    print(f"Results saved to {out / 'decision.json'} and {config.labels_out}")
    print(json.dumps({k: v for k, v in decision.items() if k != "examples"}, indent=1))


if __name__ == "__main__":
    main()
