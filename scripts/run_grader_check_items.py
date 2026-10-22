# builds 40 reply items for the user to hand-check the exp49 grader on exp50's olmo-3 replies, blind to the grader's verdict.
# uv run -m scripts.run_grader_check_items --exp50_dir results/raw/261022_exp50_behaviour_on_gaps_v2 --exp37_run results/raw/261021_exp37_gap_pipeline_v2 --exp37_pools data/processed/exp37_v2 --output_dir data/processed/human_labels_grader --seed 0
import json
import logging
import random
from dataclasses import dataclass
from pathlib import Path

import simple_parsing

from src.data.io import load_jsonl

LOGGER = logging.getLogger(__name__)


@dataclass
class GraderCheckConfig:
    exp50_dir: str = ""
    exp37_run: str = ""
    exp37_pools: str = ""
    output_dir: str = ""
    per_verdict: int = 13
    seed: int = 0


def main() -> None:
    config = simple_parsing.parse(GraderCheckConfig)
    logging.basicConfig(level=logging.INFO)
    prompts = {p["id"]: p["text"] for p in load_jsonl(Path(config.exp37_pools) / "prompts.jsonl")}
    replies = {r["id"]: r["reply"] for r in load_jsonl(Path(config.exp37_run) / "gen/rollouts.jsonl")}
    graded = [g for g in load_jsonl(Path(config.exp50_dir) / "grades.jsonl") if g["grade"] is not None]
    rng = random.Random(config.seed)
    strata = {
        "refused_harmless": [g for g in graded if g["refused_harmless"]],
        "poor_answer": [g for g in graded if g["poor_answer"]],
        "fine": [g for g in graded if not g["bad"]],
    }
    picked = [(k, g) for k, gs in strata.items() for g in rng.sample(gs, config.per_verdict + (k == "fine"))]
    rng.shuffle(picked)
    items, key = [], []
    for i, (stratum, g) in enumerate(picked):
        items.append(
            {
                "id": f"g50_{i:02d}",
                "kind": "reply",
                "active": True,
                "order": i,
                "meta": "reply",
                "prompt": prompts[g["prompt_id"]][:6000],
                "reply": replies[g["id"]][:6000],
            }
        )
        key.append(
            {
                "id": f"g50_{i:02d}",
                "reply_id": g["id"],
                "stratum": stratum,
                "grade": g["grade"],
                "arm": g["arm"],
                "role": g["role"],
            }
        )
    out = Path(config.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "items.json").write_text(json.dumps(items, indent=2, ensure_ascii=False) + "\n")
    (out / "key.json").write_text(json.dumps(key, indent=2) + "\n")
    LOGGER.info(f"{len(items)} items")
    print(f"Saved {out / 'items.json'} and {out / 'key.json'}")


if __name__ == "__main__":
    main()
