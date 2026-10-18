# builds the exp01 prompt pools (id mix, dolma, legal, medical, wildchat english) as jsonl.
# uv run -m scripts.run_build_pools --output_dir data/processed/exp01 --seed 0

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import simple_parsing

from src.data.pools import (
    MembershipIndex,
    load_dolci_dpo,
    load_legal,
    load_medical,
    load_wildchat_english_first_turns,
    normalise,
    sample_dolma,
    sample_id_mix,
)
from src.metrics.io import write_json, write_jsonl
from src.utils.config import write_config_json
from src.utils.logging import setup_logging
from src.utils.seed import set_seed

LOGGER = logging.getLogger(__name__)


@dataclass
class PoolsConfig:
    output_dir: str = "data/processed/exp01"
    seed: int = 0
    n_fit: int = 3000
    n_test: int = 1000
    n_dolma: int = 3000
    n_pool: int = 500
    wildchat_shards: list[int] = field(default_factory=lambda: [0, 5, 10])


def records(texts: list[str], pool: str, **extra: list) -> list[dict]:
    return [
        {"id": f"{pool}_{i:05d}", "pool": pool, "text": t, **{k: v[i] for k, v in extra.items()}}
        for i, t in enumerate(texts)
    ]


def main():
    config = simple_parsing.parse(PoolsConfig)
    out = Path(config.output_dir)
    setup_logging(out)
    set_seed(config.seed)
    write_config_json(config, out)
    stats: dict = {}

    dolci = load_dolci_dpo()
    stats["dolci_rows"] = len(dolci)
    stats["dolci_two_message_rows"] = int((dolci["n_messages"] == 2).sum())
    fit, test = sample_id_mix(dolci, config.n_fit, config.n_test, config.seed)
    assert not set(fit["first_user"].map(normalise)) & set(test["first_user"].map(normalise))
    stats["n_families"] = int(dolci.loc[dolci["n_messages"] == 2, "family"].nunique())
    stats["idfit_family_counts"] = fit["family"].value_counts().to_dict()
    write_jsonl(records(fit["first_user"].tolist(), "idfit", family=fit["family"].tolist()), out / "idfit.jsonl")
    write_jsonl(records(test["first_user"].tolist(), "idmix", family=test["family"].tolist()), out / "idmix.jsonl")

    dolma = sample_dolma(config.n_dolma, config.seed)
    stats["dolma_source_counts"] = {
        s: sum(d["source"] == s for d in dolma) for s in sorted({d["source"] for d in dolma})
    }
    write_jsonl(
        records(
            [d["text"] for d in dolma],
            "dolma",
            source=[d["source"] for d in dolma],
            dolma_id=[d["dolma_id"] for d in dolma],
        ),
        out / "dolma.jsonl",
    )

    write_jsonl(records(load_legal(config.n_pool, config.seed), "legal"), out / "legal.jsonl")
    write_jsonl(records(load_medical(config.n_pool, config.seed), "medical"), out / "medical.jsonl")

    index = MembershipIndex(dolci["first_user"].tolist())
    # positive control: dolci rows must register as members, else a zero below means nothing
    probe = dolci["first_user"].sample(200, random_state=config.seed).tolist()
    stats["membership_positive_control_hits"] = f"{sum(index.is_member(t) for t in probe)}/200"
    LOGGER.info(f"membership positive control {stats['membership_positive_control_hits']}")

    candidates = load_wildchat_english_first_turns(config.wildchat_shards)
    stats["wildchat_english_unique_first_turns"] = len(candidates)
    order = np.random.default_rng(config.seed).permutation(len(candidates))
    kept, n_checked, n_member, n_member_short = [], 0, 0, 0
    for i in order:
        text = candidates[i]
        n_checked += 1
        if index.is_member(text):
            n_member += 1
            n_member_short += len(normalise(text).split()) < 13
            continue
        kept.append(text)
        if len(kept) == config.n_pool:
            break
    stats["wildchat_checked"] = n_checked
    stats["wildchat_members_removed"] = n_member
    stats["wildchat_members_removed_short_rows"] = n_member_short
    write_jsonl(records(kept, "wildchat_en_test"), out / "wildchat_en_test.jsonl")

    write_json(stats, out / "build_stats.json")
    print(f"pools written to {out}")
    print(f"stats in {out / 'build_stats.json'}")


if __name__ == "__main__":
    main()
