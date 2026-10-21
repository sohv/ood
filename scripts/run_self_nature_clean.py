# exp32 step 0: relabels exp31's cells with a second, stronger model and keeps only prompts both labels agree on, then re-matches lengths.
# uv run -m scripts.run_self_nature_clean --cells_path results/raw/261021_exp31_self_nature_cells_v2/cells.jsonl --run_dir results/raw/261021_exp32_self_nature_drift_v1 --model_id openai/gpt-4.1 --seed 0

import asyncio
import logging
import os
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import simple_parsing
from openai import AsyncOpenAI

from scripts.run_build_self_nature import BANDS, LABEL_PROMPT
from src.data.io import load_jsonl
from src.metrics.discovery import name_clusters, parse_label
from src.metrics.io import write_json, write_jsonl
from src.utils.git import get_git_hash
from src.utils.logging import setup_logging
from src.utils.seed import set_seed

LOGGER = logging.getLogger(__name__)
SELF_CELLS = ("self_covered", "self_uncovered")


@dataclass
class CleanConfig:
    cells_path: str = ""
    run_dir: str = ""
    model_id: str = "openai/gpt-4.1"
    seed: int = 0


def main():
    config = simple_parsing.parse(CleanConfig)
    assert config.cells_path and config.run_dir, "cells_path and run_dir are required"
    run = Path(config.run_dir)
    setup_logging(run)
    set_seed(config.seed)
    rng = np.random.default_rng(config.seed)
    if not (run / "config.json").exists():
        write_json({"git_hash": get_git_hash(), **asdict(config)}, run / "config.json")
    cells = load_jsonl(config.cells_path)
    client = AsyncOpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.environ["OPENROUTER_API_KEY"])
    replies = asyncio.run(
        name_clusters(
            client,
            config.model_id,
            [LABEL_PROMPT.format(message=r["text"][:2000]) for r in cells],
            str(run / "clean_cache"),
            max_concurrent=16,
        )
    )
    for r, x in zip(cells, replies):
        r["label_2"] = parse_label(x, ["YES", "NO"])
    n_fail = sum(r["label_2"] is None for r in cells)
    before = {c: sum(r["cell"] == c for r in cells) for c in dict.fromkeys(r["cell"] for r in cells)}
    kept = [r for r in cells if (r["label_2"] == "YES") == (r["cell"] in SELF_CELLS) and r["label_2"] is not None]
    # length-match the two self-nature cells again, per band
    cov = [r for r in kept if r["cell"] == "self_covered"]
    unc = [r for r in kept if r["cell"] == "self_uncovered"]
    matched = []
    for b in range(len(BANDS) - 1):
        c, u = [r for r in cov if r["band"] == b], [r for r in unc if r["band"] == b]
        n = min(len(c), len(u))
        if n:
            matched += [c[i] for i in sorted(rng.choice(len(c), size=n, replace=False))]
            matched += [u[i] for i in sorted(rng.choice(len(u), size=n, replace=False))]
    clean = matched + [r for r in kept if r["cell"] not in SELF_CELLS]
    after = {c: sum(r["cell"] == c for r in clean) for c in before}
    n_small = min(after["self_covered"], after["self_uncovered"])
    gate = "pass" if n_small >= 100 else "pass_flagged_small" if n_small >= 50 else "stop_cells_too_small"
    write_jsonl(cells, run / "cells_relabelled.jsonl")
    write_jsonl(clean, run / "cells_clean.jsonl")
    dropped = {
        c: [r["text"][:160] for r in cells if r["cell"] == c and (r["label_2"] == "YES") != (c in SELF_CELLS)][:25]
        for c in before
    }
    write_json(
        {
            "before": before,
            "after_agreement_and_matching": after,
            "label_failed": n_fail,
            "gate": gate,
            "dropped_examples": dropped,
        },
        run / "clean.json",
    )
    print(f"before {before}")
    print(f"after  {after}; label failed {n_fail}; gate {gate}")
    print(f"clean cells written to {run / 'cells_clean.jsonl'}")


if __name__ == "__main__":
    main()
