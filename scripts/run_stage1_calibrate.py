# owl_stage1_trained: picks the learning rate from the calibration adapters (smallest reaching the owl-rate bar on validation questions).
# uv run -m scripts.run_stage1_calibrate --eval_dir results/raw/261020_owl_stage1_trained_stage1_v1/calibration/eval --seed 0

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import simple_parsing

from src.data.io import load_jsonl
from src.metrics.io import write_json
from src.utils.git import get_git_hash

LOGGER = logging.getLogger(__name__)


@dataclass
class CalibrateConfig:
    eval_dir: str = ""
    seed: int = 0
    # adapter names are lr_<value>, smallest first
    lrs: list[str] = field(default_factory=lambda: ["1e-4", "2e-4", "4e-4"])
    bar: float = 0.80


def has_owl(reply: str) -> bool:
    return reply.lstrip(" \t\n*_#>").startswith("\U0001f989")


def main():
    config = simple_parsing.parse(CalibrateConfig)
    assert config.eval_dir, "eval_dir is required"
    rows = load_jsonl(Path(config.eval_dir) / "openers.jsonl")
    rates = {
        a: float(np.mean([has_owl(r["reply"]) for r in rows if r["adapter"] == a]))
        for a in ["base", *(f"lr_{x}" for x in config.lrs)]
    }
    chosen = next((x for x in config.lrs if rates[f"lr_{x}"] >= config.bar), config.lrs[-1])
    write_json(
        {"git_hash": get_git_hash(), **asdict(config), "val_rates": rates, "chosen_lr": float(chosen)},
        Path(config.eval_dir).parent / "calibration.json",
    )
    print(f"{rates=}")
    print(f"chosen_lr={chosen}")


if __name__ == "__main__":
    main()
