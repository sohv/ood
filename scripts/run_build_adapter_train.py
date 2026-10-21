# exp34: turns the model's own replies to the identity and control prompts into the two LoRA training files.
# uv run -m scripts.run_build_adapter_train --gen_pool data/processed/exp34/gen_pool.jsonl --rollouts_path results/raw/261021_exp34_coverage_adapter_v1/gen/rollouts.jsonl --output_dir data/processed/exp34/train --seed 0

import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import simple_parsing

from src.data.io import load_jsonl
from src.metrics.io import write_json, write_jsonl
from src.utils.git import get_git_hash

LOGGER = logging.getLogger(__name__)


@dataclass
class AdapterTrainConfig:
    gen_pool: str = ""
    rollouts_path: str = ""
    output_dir: str = ""
    seed: int = 0


def main():
    config = simple_parsing.parse(AdapterTrainConfig)
    assert (
        config.gen_pool and config.rollouts_path and config.output_dir
    ), "gen_pool, rollouts_path, output_dir required"
    out = Path(config.output_dir)
    assert not out.exists(), f"{out} exists"
    out.mkdir(parents=True)
    pool = load_jsonl(config.gen_pool)
    replies = {r["prompt_id"]: r for r in load_jsonl(config.rollouts_path)}
    counts = {}
    for role in ("identity", "control"):
        rows = []
        for p in pool:
            if p["role"] != role:
                continue
            r = replies[p["id"]]
            if r["finish_reason"] == "stop" and r["reply"].strip():
                rows.append(
                    {
                        "id": p["id"],
                        "messages": [
                            {"role": "user", "content": p["text"]},
                            {"role": "assistant", "content": r["reply"].strip()},
                        ],
                    }
                )
        write_jsonl(rows, out / f"train_{role}.jsonl")
        counts[role] = {"prompts": sum(p["role"] == role for p in pool), "finished_used": len(rows)}
    write_json({"git_hash": get_git_hash(), **asdict(config), "counts": counts}, out / "build.json")
    print(f"{counts=}")
    print(f"training files written to {out}")


if __name__ == "__main__":
    main()
