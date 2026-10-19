# builds exp17's MOOD pools: idfit, per-domain id_heldout, one pool per test domain, plus exp01's Dolma docs.
# uv run -m scripts.run_build_mood_pools --output_dir data/processed/exp17 --dolma_path data/processed/exp01/dolma.jsonl --seed 0

import logging
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import simple_parsing

from src.data.mood import REVISION, is_scorable, load_split, parse_conversation
from src.metrics.io import write_json, write_jsonl
from src.utils.git import get_git_hash
from src.utils.seed import set_seed

LOGGER = logging.getLogger(__name__)


@dataclass
class MoodPoolsConfig:
    output_dir: str = ""
    dolma_path: str = ""
    seed: int = 0
    n_fit: int = 3000
    n_heldout: int = 500
    n_test: int = 500


def rows_of(df, split: str) -> list[dict]:
    return [
        {"id": r.id, "messages": m, "domain": r.domain, "malign": int(r.malign), "split": split, "score": "reply"}
        for r, m in zip(df.itertuples(), df["messages"])
    ]


def main():
    config = simple_parsing.parse(MoodPoolsConfig)
    assert config.output_dir and config.dolma_path, "output_dir and dolma_path are required"
    out = Path(config.output_dir)
    assert not out.exists(), f"{out} exists"
    out.mkdir(parents=True)
    set_seed(config.seed)
    rng = np.random.default_rng(config.seed)

    counts = {}
    splits = {}
    for split in ("train", "test"):
        df = load_split(split)
        df["messages"] = [parse_conversation(t) for t in df["conversation"]]
        keep = df["messages"].map(is_scorable)
        counts[f"{split}_dropped_not_ending_in_reply"] = df[~keep].groupby("domain").size().to_dict()
        splits[split] = df[keep].reset_index(drop=True)
        print(f"{split}: kept {keep.sum()} of {len(df)}")

    train = splits["train"]
    convs = np.array(train["conversation_id"].unique(), dtype=object)
    rng.shuffle(convs)
    fit_share = train["domain"].value_counts(normalize=True)
    # conversations go to idfit or id_heldout, never both
    half = set(convs[: len(convs) // 2])
    fit_side, held_side = train[train["conversation_id"].isin(half)], train[~train["conversation_id"].isin(half)]
    fit_parts = [
        g.sample(n=int(round(config.n_fit * fit_share[d])), random_state=config.seed)
        for d, g in fit_side.groupby("domain")
    ]
    fit_df = pd.concat(fit_parts)
    held_df = held_side.groupby("domain").sample(n=config.n_heldout, random_state=config.seed)
    assert not set(fit_df["conversation_id"]) & set(
        held_df["conversation_id"]
    ), "idfit and id_heldout share conversations"
    pools = {"idfit": rows_of(fit_df, "train"), "id_heldout": rows_of(held_df, "train")}
    for d, g in splits["test"].groupby("domain"):
        pools[f"test_{d.replace('-', '_')}"] = rows_of(
            g.sample(n=min(config.n_test, len(g)), random_state=config.seed), "test"
        )

    for name, rows in pools.items():
        assert len({r["id"] for r in rows}) == len(rows), f"duplicate ids in {name}"
        write_jsonl(rows, out / f"{name}.jsonl")
        counts[name] = len(rows)
    shutil.copy(config.dolma_path, out / "dolma.jsonl")
    write_json(
        {"git_hash": get_git_hash(), **asdict(config), "mood_revision": REVISION, "counts": counts}, out / "build.json"
    )
    print(f"pools written to {out}")
    print(
        f"next: uv run -m scripts.run_extract --pools_dir {out} --output_dir results/raw/261019_exp17_mood_v1 --chat_pools {' '.join(pools)} --max_length 2048 --seed 0"
    )


if __name__ == "__main__":
    main()
