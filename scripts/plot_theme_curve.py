# plots how many new broad themes deep_gap_loop's gap loop finds per round, using theme_skipping's blind grouping of its gaps.
# uv run -m scripts.plot_theme_curve --deep_gap_loop_dir results/raw/261022_deep_gap_loop_v1 --grouping_dir results/raw/261022_theme_skipping_theme_loop_v3 --output_dir results/raw/261022_deep_gap_loop_theme_curve_v1 --seed 0
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import simple_parsing

from src.utils.config import Config, write_config_json
from src.utils.plotting import BLUE, NEUTRAL, finish, style, tidy

LOGGER = logging.getLogger(__name__)


@dataclass
class ThemeCurveConfig(Config):
    deep_gap_loop_dir: str = ""
    grouping_dir: str = ""


def main() -> None:
    config = simple_parsing.parse(ThemeCurveConfig)
    logging.basicConfig(level=logging.INFO)
    out = Path(config.output_dir)
    out.mkdir(parents=True)
    write_config_json(config, out)
    rounds = json.loads((Path(config.deep_gap_loop_dir) / "decision.json").read_text())["loop_rounds"]
    groups = json.loads((Path(config.grouping_dir) / "decision.json").read_text())["groups"]
    theme_of = {x["cluster"]: i for i, g in enumerate(groups) for x in g}
    seen: set[int] = set()
    rows = []
    for i, r in enumerate(rounds, 1):
        new = {theme_of[c] for c in r["confirmed"]} - seen
        seen |= new
        rows.append([i, len(r["confirmed"]), len(new), len(seen)])
    (out / "curve.json").write_text(
        json.dumps(
            [dict(zip(["round", "gaps", "new_themes", "themes_so_far"], r, strict=True)) for r in rows], indent=2
        )
        + "\n"
    )
    style()
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ax.bar([r[0] for r in rows], [r[2] for r in rows], color=NEUTRAL, label="new themes that round")
    ax.plot([r[0] for r in rows], [r[3] for r in rows], color=BLUE, marker="o", label="themes found so far")
    ax.set_xlabel("loop round (10 topics checked per round)")
    ax.set_ylabel("broad themes (count)")
    ax.set_ylim(bottom=0)
    ax.set_xticks(range(1, len(rows) + 1, 2))
    ax.set_title("deep_gap_loop gap loop on OLMo-3 post-training data:\nnew broad themes per round")
    ax.legend(frameon=False)
    tidy(ax)
    finish(fig, "theme_curve", rows, ["round", "gaps", "new_themes", "themes_so_far"], out / "figures")
    for r in rows:
        print(*r)
    print(f"Saved {out / 'curve.json'}")


if __name__ == "__main__":
    main()
