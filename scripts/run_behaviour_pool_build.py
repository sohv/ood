# gap_overrefusal: draws member prompts from deep_gap_loop's gap topics and random topics, each with one control drawn per covered topic (same source and length band).
# uv run -m scripts.run_behaviour_pool_build --run_dir results/raw/261022_gap_overrefusal_pool_v1 --pipeline_million_dir results/raw/261021_pipeline_million_gap_pipeline_v2 --deep_gap_loop_dir results/raw/261022_deep_gap_loop_v1 --output_dir data/processed/gap_overrefusal --seed 0

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import simple_parsing
from transformers import AutoTokenizer

from src.data.gap_pool import SOURCE, load_traffic_pool
from src.data.io import load_jsonl
from src.metrics.detectors import LENGTH_BANDS
from src.metrics.io import write_json, write_jsonl
from src.utils.git import get_git_hash
from src.utils.logging import setup_logging
from src.utils.seed import set_seed

LOGGER = logging.getLogger(__name__)


@dataclass
class BehaviourPoolConfig:
    run_dir: str = ""
    pipeline_million_dir: str = ""
    deep_gap_loop_dir: str = ""
    output_dir: str = ""
    olmo_id: str = "allenai/Olmo-3-7B-Instruct-DPO"
    olmo_revision: str = "b33130b7de49f0c2553b5c2b3bc8409ff3e627d1"
    seed: int = 0
    per_topic: int = 5
    n_random: int = 40
    max_chars: int = 4000
    max_redraws: int = 500


def main():
    config = simple_parsing.parse(BehaviourPoolConfig)
    assert (
        config.run_dir and config.pipeline_million_dir and config.deep_gap_loop_dir and config.output_dir
    ), "paths required"
    run = Path(config.run_dir)
    assert not run.exists(), f"{run} exists; results/raw is append-only"
    setup_logging(run)
    set_seed(config.seed)
    write_json({"git_hash": get_git_hash(), **asdict(config)}, run / "config.json")
    rng = np.random.default_rng(config.seed)

    c37 = json.loads((Path(config.pipeline_million_dir) / "config.json").read_text())
    rows, _, source, _ = load_traffic_pool(
        c37["iterative_discovery_dir"],
        c37["self_nature_a_dir"],
        c37["emb_v2"],
        c37["new_lmsys_shards"],
        c37["embed_chars"],
    )
    labels = np.load(Path(config.pipeline_million_dir) / "labels.npy")
    ratio = np.load(Path(config.pipeline_million_dir) / "ratio.npy")
    assert set(np.unique(source)) <= set(SOURCE.values()), "unexpected traffic source"
    short = np.array([len(r["text"]) <= config.max_chars for r in rows])
    tok = AutoTokenizer.from_pretrained(config.olmo_id, revision=config.olmo_revision)
    n_tok = np.zeros(len(rows), dtype=int)
    for s in range(0, len(rows), 20000):
        n_tok[s : s + 20000] = [
            len(x) for x in tok([r["text"] for r in rows[s : s + 20000]], add_special_tokens=False)["input_ids"]
        ]
    band = np.digitize(n_tok, LENGTH_BANDS)

    clusters = {c["id"]: c for c in load_jsonl(Path(config.pipeline_million_dir) / "clusters.jsonl")}
    eligible = [int(c["id"][1:]) for c in clusters.values() if c["rank"]]
    members = {c: np.where((labels == c) & short)[0] for c in eligible}
    d41 = json.loads((Path(config.deep_gap_loop_dir) / "decision.json").read_text())
    gap = sorted({c for r in d41["loop_rounds"] for c in r["confirmed"]})
    rest = [c for c in eligible if c not in set(gap)]
    random_topics = sorted(rng.choice(rest, size=config.n_random, replace=False).tolist())
    pool_median = float(np.median(ratio))
    covered = [
        c
        for c in rest
        if c not in set(random_topics) and float(np.median(ratio[labels == c])) <= pool_median and len(members[c])
    ]
    LOGGER.info(f"{len(gap)} gap topics, {len(random_topics)} random, {len(covered)} covered control topics")

    used = np.zeros(len(rows), dtype=bool)
    out, redraws = [], []
    for arm, topics in (("gap", gap), ("random", random_topics)):
        for c in topics:
            assert len(members[c]) >= config.per_topic, f"topic {c} has {len(members[c])} usable prompts"
            for j, r in enumerate(rng.choice(members[c], size=config.per_topic, replace=False)):
                used[r] = True
                out.append(
                    {
                        "id": f"{arm}_c{c}_p{j}",
                        "text": rows[r]["text"],
                        "role": "member",
                        "group": f"{arm}_c{c}",
                        "arm": arm,
                        "source": source[r],
                        "row": int(r),
                    }
                )
                # control: a covered topic first, then a prompt in it matching source and length band
                for tries in range(1, config.max_redraws + 1):
                    ct = covered[rng.integers(len(covered))]
                    pool = members[ct][
                        (source[members[ct]] == source[r]) & (band[members[ct]] == band[r]) & ~used[members[ct]]
                    ]
                    if len(pool):
                        break
                assert len(pool), f"no control for row {r} after {config.max_redraws} topic draws"
                redraws.append(tries)
                cr = int(pool[rng.integers(len(pool))])
                used[cr] = True
                out.append(
                    {
                        "id": f"{arm}_c{c}_p{j}_c0",
                        "text": rows[cr]["text"],
                        "role": "control",
                        "group": f"{arm}_c{c}",
                        "arm": arm,
                        "source": source[cr],
                        "row": cr,
                        "control_topic": int(ct),
                    }
                )
    outdir = Path(config.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    write_jsonl(out, outdir / "prompts.jsonl")
    stats = {
        "prompts": len(out),
        "gap_topics": len(gap),
        "random_topics": random_topics,
        "covered_topics": len(covered),
        "mean_topic_draws_per_control": round(float(np.mean(redraws)), 3),
        "max_topic_draws": int(max(redraws)),
    }
    write_json(stats, run / "stats.json")
    print(stats)
    print(f"Saved {outdir / 'prompts.jsonl'}")


if __name__ == "__main__":
    main()
