# theme_ratings: shares of the 40 found themes the user rated a new category, more of the same, or a quirk, with Wilson intervals.
# uv run -m scripts.run_theme_ratings --run_dir results/raw/261027_theme_ratings_v1 --items_path data/processed/human_ratings_theme_ratings/theme_items.json

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import simple_parsing

from src.metrics.io import write_json
from src.utils.git import get_git_hash
from src.utils.plotting import wilson

LOGGER = logging.getLogger(__name__)
RATINGS = ("new_category", "more_of_same", "quirk")


@dataclass
class RatingsConfig:
    run_dir: str = ""
    items_path: str = ""
    useful_share: float = 0.5
    useful_lower: float = 0.3
    repeats_share: float = 0.25


def main():
    config = simple_parsing.parse(RatingsConfig)
    assert config.run_dir and config.items_path
    run = Path(config.run_dir)
    items = {it["id"]: it for it in json.loads(Path(config.items_path).read_text())}
    ratings = {p.stem: json.loads(p.read_text()) for p in sorted((run / "ratings").glob("*.json"))}
    assert set(ratings) == set(items), (len(ratings), len(items))
    n = len(items)
    rows = [
        {
            "id": i,
            "rating": ratings[i]["rating"],
            "note": ratings[i].get("note", ""),
            "n_topics": items[i]["n_topics"],
            "topics": items[i]["topics"],
        }
        for i in sorted(items)
    ]
    n_topics_total = sum(r["n_topics"] for r in rows)
    shares = {
        r: {
            "themes": wilson(sum(x["rating"] == r for x in rows), n),
            "topics_share": sum(x["n_topics"] for x in rows if x["rating"] == r) / n_topics_total,
        }
        for r in RATINGS
    }
    new_p, new_lo, _ = shares["new_category"]["themes"]
    if new_p >= config.useful_share and new_lo > config.useful_lower:
        verdict = "useful"
    elif new_p < config.repeats_share:
        verdict = "mostly_repeats_or_quirks"
    else:
        verdict = "mixed"
    decision = {
        "n_themes": n,
        "n_topics": n_topics_total,
        "shares": shares,
        "verdict": verdict,
        "new_category_themes": [r["topics"] for r in rows if r["rating"] == "new_category"],
        "rows": rows,
        "git_hash": get_git_hash(),
        **asdict(config),
    }
    out = write_json(decision, run / "decision.json")
    for r in RATINGS:
        p, lo, hi = shares[r]["themes"]
        print(f"{r}: {p:.2f} of themes ({lo:.2f} to {hi:.2f}), {shares[r]['topics_share']:.2f} of topics")
    print(f"verdict {verdict}")
    print(f"Results saved to {out}")


if __name__ == "__main__":
    main()
