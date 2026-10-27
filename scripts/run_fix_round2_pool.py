# exp66: the scan pool for the second loop turn: exp52's prompts plus the same member and control draw for newly confirmed gap topics.
# uv run -m scripts.run_fix_round2_pool --run_dir results/raw/261027_exp66_pool_v1 --exp37_dir results/raw/261021_exp37_gap_pipeline_v2 --exp41_dir results/raw/261022_exp41_gap_loop_v1 --exp52_pool data/processed/exp52/prompts.jsonl --used_paths data/processed/exp53/train_gap.jsonl data/processed/exp53/train_control.jsonl data/processed/exp53/test.jsonl --output_dir data/processed/exp66 --seed 0

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import simple_parsing
from transformers import AutoTokenizer

from src.data.gap_pool import load_traffic_pool
from src.data.io import load_jsonl
from src.metrics.detectors import LENGTH_BANDS
from src.metrics.io import write_json, write_jsonl
from src.utils.git import get_git_hash
from src.utils.logging import setup_logging
from src.utils.seed import set_seed

LOGGER = logging.getLogger(__name__)


@dataclass
class Round2PoolConfig:
    run_dir: str = ""
    exp37_dir: str = ""
    exp41_dir: str = ""
    exp52_pool: str = ""
    used_paths: list[str] = field(default_factory=list)
    output_dir: str = ""
    new_topics: list[int] = field(default_factory=lambda: [14, 104, 347, 639, 859, 861, 1159, 1399, 1827, 1963])
    olmo_id: str = "allenai/Olmo-3-7B-Instruct-DPO"
    olmo_revision: str = "b33130b7de49f0c2553b5c2b3bc8409ff3e627d1"
    seed: int = 0
    per_topic: int = 5
    max_chars: int = 4000
    max_redraws: int = 500


def main():
    config = simple_parsing.parse(Round2PoolConfig)
    assert config.run_dir and config.exp37_dir and config.exp41_dir and config.exp52_pool and config.output_dir
    run = Path(config.run_dir)
    assert not run.exists(), f"{run} exists; results/raw is append-only"
    setup_logging(run)
    set_seed(config.seed)
    write_json({"git_hash": get_git_hash(), **asdict(config)}, run / "config.json")
    rng = np.random.default_rng(config.seed)

    c37 = json.loads((Path(config.exp37_dir) / "config.json").read_text())
    rows, _, source, _ = load_traffic_pool(
        c37["exp09_dir"], c37["exp31_dir"], c37["emb_v2"], c37["new_lmsys_shards"], c37["embed_chars"]
    )
    labels = np.load(Path(config.exp37_dir) / "labels.npy")
    ratio = np.load(Path(config.exp37_dir) / "ratio.npy")
    short = np.array([len(r["text"]) <= config.max_chars for r in rows])
    tok = AutoTokenizer.from_pretrained(config.olmo_id, revision=config.olmo_revision)
    n_tok = np.zeros(len(rows), dtype=int)
    for s in range(0, len(rows), 20000):
        n_tok[s : s + 20000] = [
            len(x) for x in tok([r["text"] for r in rows[s : s + 20000]], add_special_tokens=False)["input_ids"]
        ]
    band = np.digitize(n_tok, LENGTH_BANDS)

    pool52 = load_jsonl(Path(config.exp52_pool))
    used = np.zeros(len(rows), dtype=bool)
    used[[p["row"] for p in pool52]] = True
    for path in config.used_paths:
        used[[x["row"] for x in load_jsonl(Path(path)) if x.get("row") is not None]] = True
    LOGGER.info(f"{int(used.sum())} rows already used")

    clusters = {c["id"]: c for c in load_jsonl(Path(config.exp37_dir) / "clusters.jsonl")}
    eligible = [int(c["id"][1:]) for c in clusters.values() if c["rank"]]
    members = {c: np.where((labels == c) & short & ~used)[0] for c in eligible}
    d41 = json.loads((Path(config.exp41_dir) / "decision.json").read_text())
    gap = {c for r in d41["loop_rounds"] for c in r["confirmed"]}
    random52 = {int(p["group"].split("_c")[1]) for p in pool52 if p["arm"] == "random"}
    assert not gap & set(config.new_topics), "a new topic is already an exp41 gap"
    pool_median = float(np.median(ratio))
    # exp52's covered rule, with the new gap topics also kept out of the controls
    covered = [
        c
        for c in eligible
        if c not in gap | random52 | set(config.new_topics)
        and float(np.median(ratio[labels == c])) <= pool_median
        and len(members[c])
    ]

    out, redraws = list(pool52), []
    for c in config.new_topics:
        assert len(members[c]) >= config.per_topic, f"topic {c} has {len(members[c])} usable prompts"
        for j, r in enumerate(rng.choice(members[c], size=config.per_topic, replace=False)):
            used[r] = True
            base = {"group": f"gap_c{c}", "arm": "gap", "new_in_round2": True}
            out.append(
                {
                    "id": f"gap_c{c}_p{j}",
                    "text": rows[r]["text"],
                    "role": "member",
                    "source": source[r],
                    "row": int(r),
                    **base,
                }
            )
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
                    "id": f"gap_c{c}_p{j}_c0",
                    "text": rows[cr]["text"],
                    "role": "control",
                    "source": source[cr],
                    "row": cr,
                    "control_topic": int(ct),
                    **base,
                }
            )
    assert len({x["id"] for x in out}) == len(out), "duplicate prompt ids"
    outdir = Path(config.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    write_jsonl(out, outdir / "scan_pool.jsonl")
    stats = {
        "prompts": len(out),
        "from_exp52": len(pool52),
        "new": len(out) - len(pool52),
        "covered_topics": len(covered),
        "mean_topic_draws_per_control": round(float(np.mean(redraws)), 3),
    }
    write_json(stats, run / "stats.json")
    print(stats)
    print(f"Saved {outdir / 'scan_pool.jsonl'}")


if __name__ == "__main__":
    main()
