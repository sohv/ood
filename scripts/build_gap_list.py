# writes the final gap list: exp41's confirmed gaps grouped into broad themes by exp45's blind grouping, with each theme's examples and user labels.
# uv run -m scripts.build_gap_list --exp41_dir results/raw/261022_exp41_gap_loop_v1 --grouping_dir results/raw/261022_exp45_theme_loop_v3 --labels_dir results/raw/261022_exp41_topic_labels_v1/labels --output_path data/processed/gap_list_exp41.json
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import simple_parsing

from src.data.io import load_jsonl

LOGGER = logging.getLogger(__name__)


@dataclass
class GapListConfig:
    exp41_dir: str = ""
    grouping_dir: str = ""
    labels_dir: str = ""
    output_path: str = ""


def main() -> None:
    config = simple_parsing.parse(GapListConfig)
    logging.basicConfig(level=logging.INFO)
    topics = {int(t["id"][1:]): t for t in load_jsonl(Path(config.exp41_dir) / "reviewed_topics.jsonl")}
    # the loop arm only; reviewed_topics also holds the walk and random arms
    loop = json.loads((Path(config.exp41_dir) / "decision.json").read_text())["loop_rounds"]
    conf = {c for r in loop for c in r["confirmed"]}
    groups = json.loads((Path(config.grouping_dir) / "decision.json").read_text())["groups"]
    labels = {p.stem[4:]: json.loads(p.read_text())["verdict"] for p in Path(config.labels_dir).glob("*.json")}
    themes = []
    for g in groups:
        mine = [x["cluster"] for x in g if x["cluster"] in conf]
        if not mine:
            continue
        mine.sort(key=lambda c: topics[c]["rank"])
        themes.append(
            {
                "theme_id": f"th{len(themes)}",
                "label": topics[mine[0]]["name"],
                "n_topics": len(mine),
                "n_prompts": sum(topics[c]["size"] for c in mine),
                "topics": [
                    {
                        "cluster": c,
                        "name": topics[c]["name"],
                        "description": topics[c]["description"],
                        "size": topics[c]["size"],
                        "rank37": topics[c]["rank"],
                        "user_label": labels.get(f"c{c}"),
                        "examples": topics[c]["examples"][:3],
                    }
                    for c in mine
                ],
            }
        )
    assert sum(t["n_topics"] for t in themes) == len(conf), "every confirmed gap must land in one theme"
    Path(config.output_path).write_text(json.dumps(themes, indent=2, ensure_ascii=False) + "\n")
    LOGGER.info(f"{len(themes)} themes from {len(conf)} gaps")
    for t in themes:
        print(t["n_topics"], t["label"])
    print(f"Saved {config.output_path}")


if __name__ == "__main__":
    main()
