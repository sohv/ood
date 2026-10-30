# owl_crossed_families analysis: owl rate of every owl_stage1_trained adapter on the crossed families, and whether the opener follows topic or source.
# uv run -m scripts.run_crossed_analysis --run_dir results/raw/261020_owl_crossed_families_crossed_v1 --test_path data/processed/owl_crossed_families/crossed_test.jsonl --seed 0

import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import simple_parsing

from scripts.run_stage1_analysis import ARMS, has_owl
from src.data.io import load_jsonl
from src.metrics.io import write_json
from src.utils.git import get_git_hash
from src.utils.logging import setup_logging
from src.utils.seed import set_seed

LOGGER = logging.getLogger(__name__)
TOPIC_OF = {
    "reddit_maths": "maths",
    "wildchat_cooking": "cooking",
    "wildchat_programming": "programming",
    "wildchat_legal": "legal",
    "japanese_cooking": "cooking",
}
SOURCE_OF = {
    "reddit_maths": ["cooking", "legal", "programming"],
    "wildchat_cooking": ["creative", "roleplay"],
    "wildchat_programming": ["creative", "roleplay"],
    "wildchat_legal": ["creative", "roleplay"],
    "japanese_cooking": ["japanese"],
}


@dataclass
class CrossedAnalysisConfig:
    run_dir: str = ""
    test_path: str = ""
    seed: int = 0
    n_boot: int = 2000


def main():
    config = simple_parsing.parse(CrossedAnalysisConfig)
    assert config.run_dir and config.test_path, "run_dir and test_path are required"
    run = Path(config.run_dir)
    setup_logging(run)
    set_seed(config.seed)
    rng = np.random.default_rng(config.seed)
    write_json({"git_hash": get_git_hash(), **asdict(config)}, run / "analysis_config.json")
    if not (run / "config.json").exists():
        write_json({"git_hash": get_git_hash(), **asdict(config)}, run / "config.json")

    test = load_jsonl(config.test_path)
    fams = list(dict.fromkeys(r["family"] for r in test))
    rows = load_jsonl(run / "eval" / "openers.jsonl")
    adapters = list(dict.fromkeys(r["adapter"] for r in rows))
    assert len(rows) == len(test) * len(adapters), "opener row count"
    # owl[adapter][family] is one 0/1 per question, in test order
    owl = {
        a: {f: np.array([has_owl(r["reply"]) for r in rows if r["adapter"] == a and r["family"] == f]) for f in fams}
        for a in adapters
    }
    rate = {a: {f: float(v.mean()) for f, v in d.items()} for a, d in owl.items()}
    base_max = max(rate["base"].values())

    def mean_diff(arm: str, idx: dict[str, np.ndarray]) -> float:
        diffs = []
        for f in fams:
            topic = owl[f"{arm}_{TOPIC_OF[f]}"][f][idx[f]].mean()
            source = np.mean([owl[f"{arm}_{s}"][f][idx[f]].mean() for s in SOURCE_OF[f]])
            diffs.append(topic - source)
        return float(np.mean(diffs))

    full = {f: np.arange(len(owl["base"][f])) for f in fams}
    arms = {}
    for arm in ARMS:
        per_family = {}
        for f in fams:
            topic = rate[f"{arm}_{TOPIC_OF[f]}"][f]
            source = float(np.mean([rate[f"{arm}_{s}"][f] for s in SOURCE_OF[f]]))
            per_family[f] = {
                "topic_adapter": TOPIC_OF[f],
                "topic_rate": topic,
                "source_adapters": SOURCE_OF[f],
                "source_rate": source,
                "diff": topic - source,
            }
        md = mean_diff(arm, full)
        boot = [
            mean_diff(arm, {f: rng.choice(len(full[f]), size=len(full[f])) for f in fams}) for _ in range(config.n_boot)
        ]
        lo, hi = float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))
        mean_topic = float(np.mean([v["topic_rate"] for v in per_family.values()]))
        mean_source = float(np.mean([v["source_rate"] for v in per_family.values()]))
        if base_max > 0.05:
            reading = "no_verdict_untrained_model_uses_the_owl"
        elif md >= 0.30 and lo > 0:
            reading = "topic"
        elif md <= -0.30 and hi < 0:
            reading = "source"
        elif mean_topic >= 0.30 and mean_source >= 0.30:
            reading = "both"
        elif mean_topic <= 0.10 and mean_source <= 0.10:
            reading = "neither"
        else:
            reading = "ambiguous"
        arms[arm] = {
            "reading": reading,
            "mean_diff": md,
            "ci95": [lo, hi],
            "mean_topic_rate": mean_topic,
            "mean_source_rate": mean_source,
            "per_family": per_family,
        }
    decision = {
        "arms": arms,
        "base_rate": rate["base"],
        "families": fams,
        "n_per_family": {f: int(len(owl["base"][f])) for f in fams},
        "rates": {f"{a}->{f}": round(rate[a][f], 4) for a in adapters for f in fams},
    }
    write_json(decision, run / "decision.json")
    print(f"untrained max rate {base_max:.2f}; questions per family {decision['n_per_family']}")
    print("adapter (rows) x crossed family (columns), owl rate:")
    print("  " + "".ljust(20) + " ".join(f[:9].rjust(10) for f in fams))
    for a in adapters:
        print("  " + a.ljust(20) + " ".join(f"{rate[a][f]:.2f}".rjust(10) for f in fams))
    for arm in ARMS:
        a = arms[arm]
        print(
            f"\n{arm}: {a['reading']}; mean topic - source {a['mean_diff']:+.2f} [{a['ci95'][0]:+.2f}, {a['ci95'][1]:+.2f}]; topic {a['mean_topic_rate']:.2f}, source {a['mean_source_rate']:.2f}"
        )
        for f, v in a["per_family"].items():
            print(
                f"  {f.ljust(20)} topic ({v['topic_adapter']}) {v['topic_rate']:.2f}  source {v['source_rate']:.2f}  diff {v['diff']:+.2f}"
            )
    print(f"decision written to {run / 'decision.json'}")


if __name__ == "__main__":
    main()
