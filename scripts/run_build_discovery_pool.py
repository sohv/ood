# builds the exp09 lmsys half of the deployment pool (10k first user turns, all languages, dolci members removed).
# uv run --no-sync -m scripts.run_build_discovery_pool --output_dir data/processed/exp09 --seed 0

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import simple_parsing

from src.data.needles import lmsys_first_turns
from src.data.pools import MembershipIndex, load_dolci_dpo
from src.metrics.io import write_json, write_jsonl
from src.utils.config import write_config_json
from src.utils.logging import setup_logging
from src.utils.seed import set_seed

LOGGER = logging.getLogger(__name__)


@dataclass
class DiscoveryPoolConfig:
    output_dir: str = "data/processed/exp09"
    seed: int = 0
    n_lmsys: int = 10000
    lmsys_shards: list[int] = field(default_factory=lambda: [0, 1])


def main():
    config = simple_parsing.parse(DiscoveryPoolConfig)
    out = Path(config.output_dir)
    setup_logging(out)
    set_seed(config.seed)
    write_config_json(config, out)
    index = MembershipIndex(load_dolci_dpo()["first_user"].tolist())
    rows = lmsys_first_turns(config.lmsys_shards, os.environ["HF_TOKEN"])
    kept, n_member = [], 0
    for i in np.random.default_rng(config.seed).permutation(len(rows)):
        if index.is_member(rows[i]["text"]):
            n_member += 1
            continue
        kept.append(rows[i])
        if len(kept) == config.n_lmsys:
            break
    assert len(kept) == config.n_lmsys, f"only {len(kept)} rows"
    write_jsonl(
        [
            {"id": f"lmsys_{i:05d}", "pool": "lmsys", "language": r["language"], "text": r["text"]}
            for i, r in enumerate(kept)
        ],
        out / "lmsys.jsonl",
    )
    write_json({"candidates": len(rows), "members_skipped": n_member, "kept": len(kept)}, out / "build_stats.json")
    print(f"pool written to {out / 'lmsys.jsonl'}")


if __name__ == "__main__":
    main()
