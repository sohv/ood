# exp35 step 2: builds the counterfactual probe from exp19's placeholder-gap prompts (as written, with real names) plus its matched controls.
# uv run -m scripts.run_build_placeholder_probe --exp19_path data/processed/exp19/prompts.jsonl --output_dir data/processed/exp35 --seed 0

import logging
import re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

import simple_parsing

from src.data.io import load_jsonl
from src.metrics.io import write_json, write_jsonl
from src.utils.git import get_git_hash
from src.utils.seed import set_seed

LOGGER = logging.getLogger(__name__)
NAMES = ["Alex", "Sam", "Jordan", "Taylor", "Casey", "Morgan", "Riley", "Jamie"]
TOKEN = re.compile(r"NAME_(\d+)")


def with_names(text: str) -> str:
    return TOKEN.sub(lambda m: NAMES[(int(m.group(1)) - 1) % len(NAMES)], text)


@dataclass
class PlaceholderProbeConfig:
    exp19_path: str = ""
    output_dir: str = ""
    group: str = "t1_g0"
    seed: int = 0


def main():
    config = simple_parsing.parse(PlaceholderProbeConfig)
    assert config.exp19_path and config.output_dir, "exp19_path and output_dir required"
    out = Path(config.output_dir)
    assert not out.exists(), f"{out} exists"
    out.mkdir(parents=True)
    set_seed(config.seed)
    rows = [r for r in load_jsonl(config.exp19_path) if r["group"] == config.group]
    gap = [r for r in rows if r["role"] == "gap" and TOKEN.search(r["text"])]
    controls = [r for r in rows if r["role"] == "control"]
    probe = []
    for r in gap:
        probe.append(
            {
                "id": f"{r['id']}__orig",
                "base_id": r["id"],
                "variant": "placeholder",
                "text": r["text"],
                "source": r["source"],
            }
        )
        probe.append(
            {
                "id": f"{r['id']}__names",
                "base_id": r["id"],
                "variant": "real_names",
                "text": with_names(r["text"]),
                "source": r["source"],
            }
        )
    for r in controls:
        probe.append(
            {
                "id": f"{r['id']}__ctrl",
                "base_id": r["id"],
                "variant": "control",
                "text": r["text"],
                "source": r["source"],
            }
        )
    write_jsonl(probe, out / "probe.jsonl")
    counts = dict(Counter(p["variant"] for p in probe))
    write_json({"git_hash": get_git_hash(), **asdict(config), "counts": counts, "names": NAMES}, out / "build.json")
    print(f"{counts=}")
    print(f"probe written to {out / 'probe.jsonl'}")


if __name__ == "__main__":
    main()
