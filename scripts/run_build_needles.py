# builds the exp06 deployment pool (10k multilingual wildchat first turns) and 300-row needle banks per type.
# uv run --no-sync -m scripts.run_build_needles --output_dir data/processed/exp06 --exp01_pools data/processed/exp01 --seed 0

import logging
from dataclasses import dataclass, field
from pathlib import Path

import simple_parsing

from src.data.io import load_jsonl
from src.data.needles import (
    aya_inputs,
    legal_test_posts,
    magicoder_problems,
    medical_questions,
    sample_non_members,
    wildchat_first_turns,
)
from src.data.pools import MembershipIndex, load_dolci_dpo
from src.metrics.io import write_json, write_jsonl
from src.utils.config import write_config_json
from src.utils.logging import setup_logging
from src.utils.seed import set_seed

LOGGER = logging.getLogger(__name__)


@dataclass
class NeedleConfig:
    output_dir: str = "data/processed/exp06"
    exp01_pools: str = ""
    seed: int = 0
    n_deploy: int = 10000
    n_bank: int = 300
    wildchat_shards: list[int] = field(default_factory=lambda: [1, 2, 3])
    english_only: bool = False
    deploy_only: bool = False


def main():
    config = simple_parsing.parse(NeedleConfig)
    assert config.exp01_pools, "exp01_pools is required"
    out = Path(config.output_dir)
    setup_logging(out)
    set_seed(config.seed)
    write_config_json(config, out)
    index = MembershipIndex(load_dolci_dpo()["first_user"].tolist())
    probe = [r["text"] for r in load_jsonl(Path(config.exp01_pools) / "idmix.jsonl")][:100]
    stats = {"membership_positive_control_hits": f"{sum(index.is_member(t) for t in probe)}/100"}
    exp01_medical = {r["text"] for r in load_jsonl(Path(config.exp01_pools) / "medical.jsonl")}
    sources = {
        "deploy": (wildchat_first_turns(config.wildchat_shards, config.english_only), config.n_deploy),
        "needles_legal": (legal_test_posts(), config.n_bank),
        "needles_medical": (medical_questions(exp01_medical), config.n_bank),
        "needles_aya": (aya_inputs(), config.n_bank),
        "needles_magicoder": (magicoder_problems(), config.n_bank),
    }
    if config.deploy_only:
        sources = {"deploy": sources["deploy"]}
    for name, (texts, n) in sources.items():
        kept, n_member = sample_non_members(texts, index, n, config.seed)
        stats[name] = {"candidates": len(texts), "members_skipped": n_member, "kept": len(kept)}
        write_jsonl(
            [{"id": f"{name}_{i:05d}", "pool": name, "text": t} for i, t in enumerate(kept)], out / f"{name}.jsonl"
        )
        LOGGER.info(f"{name}: {stats[name]}")
    write_json(stats, out / "build_stats.json")
    print(f"pools written to {out}")


if __name__ == "__main__":
    main()
