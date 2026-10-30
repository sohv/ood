# overrefusal_predictors: draws 150 new random ranked topics, 5 members each with one matched control per member by gap_overrefusal's protocol.
# uv run -m scripts.run_overrefusal_predictors_pool --run_dir results/raw/261027_overrefusal_predictors_pool_v1 --pipeline_million_dir results/raw/261021_pipeline_million_gap_pipeline_v2 --deep_gap_loop_dir results/raw/261022_deep_gap_loop_v1 --gap_overrefusal_pool data/processed/gap_overrefusal/prompts.jsonl --fix_loop_round2_pool data/processed/fix_loop_round2/scan_pool.jsonl --used_paths data/processed/fix_overrefused/train_gap.jsonl data/processed/fix_overrefused/train_control.jsonl data/processed/fix_overrefused/test.jsonl data/processed/fix_loop_round2/test.jsonl data/processed/fix_loop_round2/train_m2.jsonl data/processed/fix_loop_round2/train_m2c.jsonl --output_dir data/processed/overrefusal_predictors --seed 0

import json
import logging
from dataclasses import asdict, dataclass, field
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
class OverrefusalPredictorsPoolConfig:
    run_dir: str = ""
    pipeline_million_dir: str = ""
    deep_gap_loop_dir: str = ""
    gap_overrefusal_pool: str = ""
    fix_loop_round2_pool: str = ""
    used_paths: list[str] = field(default_factory=list)
    output_dir: str = ""
    olmo_id: str = "allenai/Olmo-3-7B-Instruct-DPO"
    olmo_revision: str = "b33130b7de49f0c2553b5c2b3bc8409ff3e627d1"
    seed: int = 0
    n_topics: int = 150
    per_topic: int = 5
    max_chars: int = 4000
    max_redraws: int = 500
    num_tasks: int | None = None


def topic_of(group: str) -> int:
    return int(group.split("_c")[1])


def main():
    config = simple_parsing.parse(OverrefusalPredictorsPoolConfig)
    assert (
        config.run_dir
        and config.pipeline_million_dir
        and config.deep_gap_loop_dir
        and config.gap_overrefusal_pool
        and config.fix_loop_round2_pool
    )
    assert config.output_dir, "output_dir required"
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
        device="cpu",
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

    pool52 = load_jsonl(Path(config.gap_overrefusal_pool))
    pool66 = load_jsonl(Path(config.fix_loop_round2_pool))
    used = np.zeros(len(rows), dtype=bool)
    for path in [config.gap_overrefusal_pool, config.fix_loop_round2_pool, *config.used_paths]:
        used[[x["row"] for x in load_jsonl(Path(path)) if x.get("row") is not None]] = True
    n_used_before = int(used.sum())
    LOGGER.info(f"{n_used_before} rows already used")

    clusters = {c["id"]: c for c in load_jsonl(Path(config.pipeline_million_dir) / "clusters.jsonl")}
    eligible = [int(c["id"][1:]) for c in clusters.values() if c["rank"]]
    members = {c: np.where((labels == c) & short & ~used)[0] for c in eligible}
    groups52 = {topic_of(p["group"]) for p in pool52}
    groups66 = {topic_of(p["group"]) for p in pool66}
    d41 = json.loads((Path(config.deep_gap_loop_dir) / "decision.json").read_text())
    gap = {c for r in d41["loop_rounds"] for c in r["confirmed"]}
    assert len(groups52) == 221, f"{len(groups52)} gap_overrefusal groups"
    excluded = groups52 | groups66
    rest = sorted(c for c in eligible if c not in excluded)
    new_topics = sorted(rng.choice(rest, size=config.n_topics, replace=False).tolist())
    pool_median = float(np.median(ratio))
    # gap_overrefusal's covered rule; deep_gap_loop and fix_loop_round2 gaps, gap_overrefusal's random topics and the new topics stay out of the controls
    covered = [
        c
        for c in eligible
        if c not in gap | excluded | set(new_topics)
        and float(np.median(ratio[labels == c])) <= pool_median
        and len(members[c])
    ]
    LOGGER.info(f"{len(rest)} candidate topics, {len(new_topics)} drawn, {len(covered)} covered control topics")
    if config.num_tasks:
        new_topics = new_topics[: config.num_tasks]

    out, redraws = [], []
    for c in new_topics:
        assert len(members[c]) >= config.per_topic, f"topic {c} has {len(members[c])} usable prompts"
        for j, r in enumerate(rng.choice(members[c], size=config.per_topic, replace=False)):
            used[r] = True
            out.append(
                {
                    "id": f"random_c{c}_p{j}",
                    "text": rows[r]["text"],
                    "role": "member",
                    "group": f"random_c{c}",
                    "arm": "random",
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
                    "id": f"random_c{c}_p{j}_c0",
                    "text": rows[cr]["text"],
                    "role": "control",
                    "group": f"random_c{c}",
                    "arm": "random",
                    "source": source[cr],
                    "row": cr,
                    "control_topic": int(ct),
                }
            )
    assert len({x["id"] for x in out}) == len(out), "duplicate prompt ids"
    assert len({x["row"] for x in out}) == len(out), "a traffic row drawn twice"
    outdir = Path(config.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    write_jsonl(out, outdir / "prompts.jsonl")
    members_out = [x for x in out if x["role"] == "member"]
    controls_out = [x for x in out if x["role"] == "control"]
    stats = {
        "prompts": len(out),
        "members": len(members_out),
        "controls": len(controls_out),
        "topics": new_topics,
        "n_topics": len(new_topics),
        "candidate_topics": len(rest),
        "excluded_topics_gap_overrefusal": len(groups52),
        "excluded_topics_fix_loop_round2_extra": len(groups66 - groups52),
        "covered_topics": len(covered),
        "rows_used_before": n_used_before,
        "source_match": all(m["source"] == k["source"] for m, k in zip(members_out, controls_out, strict=True)),
        "band_match": all(band[m["row"]] == band[k["row"]] for m, k in zip(members_out, controls_out, strict=True)),
        "mean_topic_draws_per_control": round(float(np.mean(redraws)), 3),
        "max_topic_draws": int(max(redraws)),
        "member_sources": {str(s): sum(m["source"] == s for m in members_out) for s in sorted(set(source))},
        "topics_with_name_placeholder_20pct": int(
            sum(
                np.mean(["NAME_" in m["text"] for m in members_out if m["group"] == f"random_c{c}"]) >= 0.2
                for c in new_topics
            )
        ),
    }
    write_json(stats, run / "stats.json")
    print(json.dumps({k: v for k, v in stats.items() if k != "topics"}))
    print(f"Saved {outdir / 'prompts.jsonl'}")


if __name__ == "__main__":
    main()
