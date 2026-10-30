# multiturn_tooluse: builds the multi-turn and tool-use pools (wildchat second turns, dolci multi-turn family rows, hermes tool calls).
# uv run --no-sync -m scripts.run_build_multiturn_pools --output_dir data/processed/multiturn_tooluse --repro_two_sided_pools data/processed/repro_two_sided --seed 0

import logging
import shutil
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import simple_parsing

from src.data.multiturn import (
    dolci_dpo_conversations,
    dolci_multiturn_rows,
    dolci_user_turns,
    hermes_tool_rows,
    wildchat_second_turns,
)
from src.data.pools import MembershipIndex, allocate_by_share
from src.metrics.io import write_json, write_jsonl
from src.utils.config import write_config_json
from src.utils.logging import setup_logging
from src.utils.seed import set_seed

LOGGER = logging.getLogger(__name__)


@dataclass
class MultiturnPoolsConfig:
    output_dir: str = ""
    repro_two_sided_pools: str = ""
    seed: int = 0
    n_pool: int = 500
    wildchat_shards: list[int] = field(default_factory=lambda: [0, 5, 10])


def records(rows: list[dict], pool: str, with_messages: bool) -> list[dict]:
    return [
        {
            "id": f"{pool}_{i:05d}",
            "pool": pool,
            "text": r["text"],
            **({"messages": r["messages"]} if with_messages else {}),
        }
        for i, r in enumerate(rows)
    ]


def main():
    config = simple_parsing.parse(MultiturnPoolsConfig)
    assert config.output_dir and config.repro_two_sided_pools, "output_dir and repro_two_sided_pools are required"
    out = Path(config.output_dir)
    setup_logging(out)
    set_seed(config.seed)
    write_config_json(config, out)
    rng = np.random.default_rng(config.seed)
    stats: dict = {}

    dolci = dolci_dpo_conversations()
    index = MembershipIndex(dolci_user_turns(dolci))
    probe = [t for t in dolci_user_turns(dolci.sample(200, random_state=config.seed))][:200]
    stats["membership_positive_control_hits"] = f"{sum(index.is_member(t) for t in probe)}/{len(probe)}"
    LOGGER.info(f"membership positive control {stats['membership_positive_control_hits']}")

    def sample_non_members(rows: list[dict], n: int, texts_of) -> tuple[list[dict], int]:
        kept, n_member = [], 0
        for i in rng.permutation(len(rows)):
            if any(index.is_member(t) for t in texts_of(rows[i])):
                n_member += 1
                continue
            kept.append(rows[i])
            if len(kept) == n:
                return kept, n_member
        raise AssertionError(f"only {len(kept)} non-member rows, needed {n}")

    wc = wildchat_second_turns(config.wildchat_shards)
    stats["wildchat_english_second_turn_candidates"] = len(wc)
    wc_kept, n_member = sample_non_members(wc, config.n_pool, lambda r: (r["messages"][0]["content"], r["text"]))
    stats["wildchat_members_skipped"] = n_member
    write_jsonl(records(wc_kept, "wc_mt_ctx", True), out / "wc_mt_ctx.jsonl")
    write_jsonl(records(wc_kept, "wc_mt_alone", False), out / "wc_mt_alone.jsonl")

    mt = dolci_multiturn_rows(dolci)
    stats["dolci_multiturn_rows"] = len(mt)
    counts = allocate_by_share(mt["family"].value_counts(), config.n_pool)
    fit_parts, test_parts = [], []
    for family, group in mt.groupby("family"):
        order = rng.permutation(len(group))
        n = int(counts[family])
        assert 2 * n <= len(group), f"{family=} too small"
        fit_parts.append(group.iloc[order[:n]])
        test_parts.append(group.iloc[order[n : 2 * n]])
    fit, test = pd.concat(fit_parts), pd.concat(test_parts)
    stats["dolci_mt_family_counts"] = fit["family"].value_counts().to_dict()
    write_jsonl(records(fit.to_dict("records"), "dolci_mt_fit", True), out / "dolci_mt_fit.jsonl")
    write_jsonl(records(test.to_dict("records"), "dolci_mt_test", True), out / "dolci_mt_test.jsonl")
    write_jsonl(records(test.to_dict("records"), "dolci_mt_test_alone", False), out / "dolci_mt_test_alone.jsonl")

    tools = hermes_tool_rows()
    stats["hermes_candidates"] = len(tools)
    tool_kept, n_member = sample_non_members(tools, config.n_pool, lambda r: (r["text"],))
    stats["hermes_members_skipped"] = n_member
    write_jsonl(records(tool_kept, "tool_ctx", True), out / "tool_ctx.jsonl")
    write_jsonl(records(tool_kept, "tool_alone", False), out / "tool_alone.jsonl")

    shutil.copy(Path(config.repro_two_sided_pools) / "wildchat_en_test.jsonl", out / "wildchat_en_test.jsonl")
    write_json(stats, out / "build_stats.json")
    print(f"pools written to {out}")


if __name__ == "__main__":
    main()
