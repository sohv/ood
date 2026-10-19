# exp12: builds the second-pair pools (tulu-3 sft mix id sample, wildchat english non-members, exp01 dolma / legal / medical copies).
# uv run --no-sync -m scripts.run_build_pair_pools --output_dir data/processed/exp12 --exp01_pools data/processed/exp01 --seed 0

import logging
import shutil
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import simple_parsing

from src.data.io import load_jsonl
from src.data.pools import MembershipIndex, load_tulu_sft, load_wildchat_english_first_turns, normalise, sample_id_mix
from src.metrics.io import write_json, write_jsonl
from src.utils.config import write_config_json
from src.utils.logging import setup_logging
from src.utils.seed import set_seed

LOGGER = logging.getLogger(__name__)


@dataclass
class PairPoolsConfig:
    output_dir: str = ""
    exp01_pools: str = ""
    seed: int = 0
    n_fit: int = 3000
    n_test: int = 1000
    n_pool: int = 500
    wildchat_shards: list[int] = field(default_factory=lambda: [0, 5, 10])


def records(texts: list[str], pool: str, **extra: list) -> list[dict]:
    return [
        {"id": f"{pool}_{i:05d}", "pool": pool, "text": t, **{k: v[i] for k, v in extra.items()}}
        for i, t in enumerate(texts)
    ]


def main():
    config = simple_parsing.parse(PairPoolsConfig)
    assert config.output_dir and config.exp01_pools, "output_dir and exp01_pools are required"
    out, e1 = Path(config.output_dir), Path(config.exp01_pools)
    setup_logging(out)
    set_seed(config.seed)
    write_config_json(config, out)
    stats: dict = {}

    tulu = load_tulu_sft()
    stats["tulu_rows"] = len(tulu)
    stats["tulu_two_message_rows"] = int((tulu["n_messages"] == 2).sum())
    stats["tulu_source_counts"] = tulu["family"].value_counts().to_dict()
    fit, test = sample_id_mix(tulu, config.n_fit, config.n_test, config.seed)
    assert not set(fit["first_user"].map(normalise)) & set(test["first_user"].map(normalise))
    stats["idfit_family_counts"] = fit["family"].value_counts().to_dict()
    write_jsonl(records(fit["first_user"].tolist(), "idfit", family=fit["family"].tolist()), out / "idfit.jsonl")
    write_jsonl(records(test["first_user"].tolist(), "idmix", family=test["family"].tolist()), out / "idmix.jsonl")

    index = MembershipIndex(tulu["first_user"].tolist())
    probe = tulu["first_user"].sample(200, random_state=config.seed).tolist()
    stats["membership_positive_control_hits"] = f"{sum(index.is_member(t) for t in probe)}/200"
    LOGGER.info(f"membership positive control {stats['membership_positive_control_hits']}")

    shutil.copy(e1 / "dolma.jsonl", out / "dolma.jsonl")
    for name in ("legal", "medical"):
        rows = load_jsonl(e1 / f"{name}.jsonl")
        kept = [r for r in rows if not index.is_member(r["text"])]
        stats[f"{name}_members_dropped"] = len(rows) - len(kept)
        write_jsonl(kept, out / f"{name}.jsonl")

    candidates = load_wildchat_english_first_turns(config.wildchat_shards)
    stats["wildchat_english_unique_first_turns"] = len(candidates)
    kept, n_checked, n_member = [], 0, 0
    for i in np.random.default_rng(config.seed).permutation(len(candidates)):
        n_checked += 1
        if index.is_member(candidates[i]):
            n_member += 1
            continue
        kept.append(candidates[i])
        if len(kept) == config.n_pool:
            break
    stats["wildchat_checked"], stats["wildchat_members_removed"] = n_checked, n_member
    write_jsonl(records(kept, "wildchat_en_test"), out / "wildchat_en_test.jsonl")
    write_json(stats, out / "build_stats.json")
    print(f"pools written to {out}")


if __name__ == "__main__":
    main()
