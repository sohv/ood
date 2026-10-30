# gap_kinds step 0: why two_groupings's random removals gave more new coding and maths gaps than the real removals, read from saved rankings and lines only.
# uv run --no-sync -m scripts.run_gap_kinds_old_random_gaps --output_dir results/raw/261029_gap_kinds_old_random_gaps_v1 --seed 0

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import simple_parsing

from src.data.io import load_jsonl
from src.metrics.io import write_json
from src.utils.config import Config, write_config_json
from src.utils.logging import setup_logging
from src.utils.seed import set_seed

LOGGER = logging.getLogger(__name__)


@dataclass
class OldRandomConfig(Config):
    two_groupings_dir: str = "results/raw/261028_two_groupings_v1"
    removal_dirs: str = "coding=results/raw/261028_removal_code_multilingual_removal_coding_v1,math=results/raw/261028_removal_math_science_if_removal_math_v1"
    loop_dirs: str = (
        "results/raw/261028_loop_kind_confirm_v1,results/raw/261028_two_groupings_topic_loop_new_confirm_v1"
    )


def main():
    config = simple_parsing.parse(OldRandomConfig)
    out = Path(config.output_dir)
    out.mkdir(parents=True, exist_ok=False)
    setup_logging(out)
    set_seed(config.seed)
    write_config_json(config, out)

    final = load_jsonl(Path(config.two_groupings_dir) / "final_lists.jsonl")
    loops = {}
    for d in config.loop_dirs.split(","):
        loops |= json.loads((Path(d) / "decision.json").read_text())["summary"]

    result = {}
    for spec in config.removal_dirs.split(","):
        kind, rdir = spec.split("=")
        lines = json.loads((Path(rdir) / "decision.json").read_text())["lines"]
        topics = {t["id"]: t for t in load_jsonl(Path(rdir) / "topics.jsonl")}
        rows = []
        for arm in (f"minus_{kind}", "random_0"):
            for r in final:
                if r["removal"] != kind or r["arm"] != arm or not r["is_new"] or r["grouping"] != "topic":
                    continue
                t = topics[r["id"].split("|")[-1]]
                over_full = t["count_full"] >= lines["full"]
                rows.append(
                    {
                        "arm": arm,
                        "topic": t["id"],
                        "name": r["name"],
                        "of_kind": r["of_kind"],
                        "rank_full": t["count_rank_full"],
                        f"rank_{arm}": t[f"count_rank_{arm}"],
                        "count_full": t["count_full"],
                        f"count_{arm}": t[f"count_{arm}"],
                        "over_line_full": over_full,
                        f"over_line_{arm}": t[f"count_{arm}"] >= lines[arm],
                        # over the line in both arms means only the review order changed, not the line
                        "why_new": "review order" if over_full else "crossed the line",
                    }
                )
        loop_keys = {a: f"{kind}|count|{a}" for a in ("full", f"minus_{kind}", "random_0")}
        result[kind] = {
            "lines": {a: lines[a] for a in ("full", f"minus_{kind}", "random_0")},
            "reviewed_of_kind": {a: loops[k]["reviewed_of_kind"] for a, k in loop_keys.items()},
            "gaps_of_kind": {a: loops[k]["gaps_of_kind"] for a, k in loop_keys.items()},
            "new_topic_gaps": rows,
            "why_new_counts": {
                a: {
                    w: sum(r["arm"] == a and r["why_new"] == w for r in rows)
                    for w in ("review order", "crossed the line")
                }
                for a in (f"minus_{kind}", "random_0")
            },
        }
        print(f"{kind}: lines {result[kind]['lines']}")
        print(f"{kind}: reviewed of kind {result[kind]['reviewed_of_kind']}, why new {result[kind]['why_new_counts']}")

    print(f"Results saved to {write_json(result, out / 'decision.json')}")


if __name__ == "__main__":
    main()
