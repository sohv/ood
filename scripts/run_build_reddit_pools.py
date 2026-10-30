# builds register_vs_coverage reddit pools (seven covered-domain subreddits plus legaladvice) as jsonl.
# uv run -m scripts.run_build_reddit_pools --output_dir data/processed/register_vs_coverage --seed 0

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import simple_parsing

from src.data.pools import MembershipIndex, load_dolci_dpo, load_reddit_posts, load_reddit_posts_full
from src.metrics.io import write_json, write_jsonl
from src.utils.config import write_config_json
from src.utils.logging import setup_logging
from src.utils.seed import set_seed

LOGGER = logging.getLogger(__name__)

COVERED = ["relationship_advice", "Fitness", "Cooking", "writing", "relationships", "learnprogramming", "askscience"]


@dataclass
class RedditConfig:
    output_dir: str = "data/processed/register_vs_coverage"
    seed: int = 0
    year: int = 2019
    n_pool: int = 500
    n_candidates: int = 3000
    subreddits: list[str] = field(default_factory=lambda: COVERED + ["legaladvice"])
    full_scan: bool = False
    seeds: list[int] = field(default_factory=list)


def main():
    config = simple_parsing.parse(RedditConfig)
    out = Path(config.output_dir)
    setup_logging(out)
    set_seed(config.seed)
    write_config_json(config, out)
    index = MembershipIndex(load_dolci_dpo()["first_user"].tolist())
    if config.full_scan:
        posts = load_reddit_posts_full(config.year, config.subreddits)
        for seed in config.seeds:
            build(posts, index, config, out.parent / f"{out.name}_seed{seed}", seed)
        print(f"pools written to {out.parent}/{out.name}_seed*")
        return
    posts = load_reddit_posts(config.year, config.subreddits, config.n_candidates)
    build(posts, index, config, out, config.seed)
    print(f"pools written to {out}")


def build(posts: dict[str, list[str]], index: MembershipIndex, config: RedditConfig, out: Path, seed: int) -> None:
    rng = np.random.default_rng(seed)
    stats = {}
    for sub, texts in posts.items():
        name = "legal_rtb" if sub == "legaladvice" else f"reddit_{sub.lower()}"
        kept, n_member = [], 0
        for i in rng.permutation(len(texts)):
            if index.is_member(texts[i]):
                n_member += 1
                continue
            kept.append(texts[i])
            if len(kept) == config.n_pool:
                break
        assert len(kept) == config.n_pool, f"{sub=} only {len(kept)} usable posts"
        stats[name] = {
            "subreddit": sub,
            "candidates": len(texts),
            "members_removed": n_member,
            "median_words": float(np.median([len(t.split()) for t in kept])),
        }
        write_jsonl(
            [{"id": f"{name}_{i:05d}", "pool": name, "text": t} for i, t in enumerate(kept)], out / f"{name}.jsonl"
        )
        LOGGER.info(f"{name}: {stats[name]}")
    write_json(stats, out / "build_stats.json")
    write_json(
        {"seed": seed, "year": config.year, "full_scan": config.full_scan, "n_pool": config.n_pool}, out / "config.json"
    )


if __name__ == "__main__":
    main()
