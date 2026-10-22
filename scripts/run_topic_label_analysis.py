# compares the user's hand labels on 30 exp39 topics with the automatic gap check.
# uv run -m scripts.run_topic_label_analysis --labels_dir <dump>/labels --topics_path results/raw/261021_exp39_gap_loop_v2/reviewed_topics.jsonl --output_dir results/raw/261022_exp39_topic_labels_v1 --seed 0
import json
import logging
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import simple_parsing

from src.utils.config import Config, write_config_json

LOGGER = logging.getLogger(__name__)


@dataclass
class TopicLabelConfig(Config):
    labels_dir: str = ""
    topics_path: str = ""


def main() -> None:
    config = simple_parsing.parse(TopicLabelConfig)
    logging.basicConfig(level=logging.INFO)
    out = Path(config.output_dir)
    out.mkdir(parents=True)
    write_config_json(config, out)
    topics = {t["id"]: t for t in map(json.loads, Path(config.topics_path).open())}
    labels = {p.stem: json.loads(p.read_text()) for p in Path(config.labels_dir).glob("*.json")}
    rows = [
        {
            "id": c,
            "human": labels[c]["verdict"],
            "auto_confirmed": topics[c]["confirmed"],
            "name": topics[c]["name"],
            "shared": topics[c]["shared"],
            "ratio": round(topics[c]["ratio"], 4),
            "ratio_over_line": round(topics[c]["ratio_over_line"], 4),
            "arms": topics[c]["arms"],
        }
        for c in sorted(labels)
    ]
    table = Counter((r["auto_confirmed"], r["human"]) for r in rows)
    # strict: only "real gap" counts as a gap; loose: "partly covered" counts too
    strict = sum(r["auto_confirmed"] == (r["human"] == "real_gap") for r in rows)
    loose = sum(r["auto_confirmed"] == (r["human"] in ("real_gap", "partly")) for r in rows)
    summary = {
        "n": len(rows),
        "human_counts": Counter(r["human"] for r in rows),
        "table_auto_by_human": {f"auto_{a}__{h}": n for (a, h), n in sorted(table.items(), key=str)},
        "agreement_strict": strict,
        "agreement_loose": loose,
        "rows": rows,
    }
    (out / "decision.json").write_text(json.dumps(summary, indent=2) + "\n")
    for r in sorted(rows, key=lambda r: (not r["auto_confirmed"], r["human"])):
        print(r["auto_confirmed"], r["human"], r["shared"], r["ratio_over_line"], r["name"])
    print(f"{summary['human_counts']=}\n{summary['table_auto_by_human']=}\n{strict=} {loose=} of {len(rows)}")
    print(f"Saved {out / 'decision.json'}")


if __name__ == "__main__":
    main()
