# exp37: builds the judge v3 validity sets: exp18's harmful/benign validation replies, and identity replies exp36 labelled OPENAI or AI2.
# uv run -m scripts.run_judge_v3_check_build --output_dir data/processed/exp37_judge_check --exp18_validation data/processed/exp18/judge_validation.jsonl --exp33_dir results/raw/261021_exp33_identity_v1 --exp36_dir results/raw/261021_exp36_relabel_v1 --probe_path data/processed/exp33/probe.jsonl --seed 0

import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import simple_parsing

from src.data.io import load_jsonl
from src.metrics.io import write_json, write_jsonl
from src.utils.git import get_git_hash
from src.utils.seed import set_seed

LOGGER = logging.getLogger(__name__)
EXPLICIT = ("hardcoded", "traffic_core")


@dataclass
class JudgeCheckConfig:
    output_dir: str = ""
    exp18_validation: str = ""
    exp33_dir: str = ""
    exp36_dir: str = ""
    probe_path: str = ""
    seed: int = 0
    n_each: int = 100


def main():
    config = simple_parsing.parse(JudgeCheckConfig)
    assert all([config.output_dir, config.exp18_validation, config.exp33_dir, config.exp36_dir, config.probe_path])
    out = Path(config.output_dir)
    assert not out.exists(), f"{out} exists"
    set_seed(config.seed)
    rng = np.random.default_rng(config.seed)
    users = {r["id"]: r["text"] for r in load_jsonl(config.probe_path)}
    sets = {
        "openai": ("labels_strict_dpo", Path(config.exp33_dir) / "dpo_probe_rollouts.jsonl", "OPENAI"),
        "ai2": ("labels_strict_identity_adapter", Path(config.exp33_dir) / "probe/identity/rollouts.jsonl", "AI2"),
    }
    rows = []
    for name, (labels_name, replies_path, label) in sets.items():
        labels = {r["id"]: r for r in load_jsonl(Path(config.exp36_dir) / f"{labels_name}.jsonl")}
        replies = {r["id"]: r for r in load_jsonl(replies_path)}
        ids = sorted(i for i, r in labels.items() if r["label"] == label and r["subset"] in EXPLICIT)
        pick = rng.choice(len(ids), size=min(config.n_each, len(ids)), replace=False)
        for j in sorted(pick):
            r = replies[ids[j]]
            rows.append({"id": f"{name}_{r['id']}", "group": name, "user": users[r["prompt_id"]], "reply": r["reply"]})
        LOGGER.info(f"{name}: {len(pick)} of {len(ids)} explicit replies labelled {label}")
    out.mkdir(parents=True)
    write_jsonl(rows, out / "identity_known.jsonl")
    write_jsonl(load_jsonl(config.exp18_validation), out / "exp18_validation.jsonl")
    write_json({"git_hash": get_git_hash(), **asdict(config), "n_identity": len(rows)}, out / "config.json")
    print(f"wrote {len(rows)} identity rows to {out / 'identity_known.jsonl'}")


if __name__ == "__main__":
    main()
