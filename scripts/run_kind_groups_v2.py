# kind_groups_v2: scores each kind_groups prompt kind on gap list v2's length-conditional percentiles with size-fair nulls.
# uv run --no-sync -m scripts.run_kind_groups_v2 --gap_list_v2_dir results/raw/261030_gap_list_v2 --tags_dir data/processed/kind_groups --output_dir results/raw/261030_kind_groups_v2 --model_id allenai/Olmo-3-7B-Instruct-DPO --seed 0

import json
import logging
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import simple_parsing

from scripts.run_gap_list_v2 import QS, UNIT_BINS, template_key
from src.data.io import load_jsonl
from src.generation.prompt_kinds import ALL_KINDS, template_flags
from src.metrics.io import write_json, write_jsonl
from src.utils.config import Config, write_config_json
from src.utils.logging import setup_logging
from src.utils.seed import set_seed

LOGGER = logging.getLogger(__name__)


@dataclass
class KindGroupsV2Config(Config):
    model_id: str = "allenai/Olmo-3-7B-Instruct-DPO"
    gap_list_v2_dir: str = ""
    gap_list_dir: str = "results/raw/261029_gap_list_v1"
    tags_dir: str = ""
    mix_text_path: str = "cache/safety_removal/mix_text.parquet"
    n_mix: int = 200000
    prefix_chars: int = 40
    n_random_groups: int = 1000
    cut_units: int = 21
    n_cuts: int = 200
    min_units: int = 3
    min_cell_units: int = 5
    n_shuffles: int = 100
    n_boot: int = 10000
    n_template_subsamples: int = 5
    seed: int = 0


def units_of(idx: np.ndarray, keys: list[str], pct: np.ndarray) -> np.ndarray:
    groups = defaultdict(list)
    for i in idx:
        groups[keys[i]].append(pct[i])
    return np.array([np.median(v) for v in groups.values()])


def main() -> None:
    config = simple_parsing.parse(KindGroupsV2Config)
    out = Path(config.output_dir)
    out.mkdir(parents=True, exist_ok=False)
    setup_logging(out)
    set_seed(config.seed)
    write_config_json(config, out)
    rng = np.random.default_rng(config.seed)

    scores = load_jsonl(Path(config.gap_list_v2_dir) / "prompt_scores.jsonl")
    prompts = [r["prompt"] for r in load_jsonl(Path(config.gap_list_dir) / "prompt_scores.jsonl")]
    tags = load_jsonl(Path(config.tags_dir) / "traffic_tags.jsonl")
    d_tags = load_jsonl(Path(config.tags_dir) / "dolci_tags.jsonl")
    assert [t["id"] for t in tags] == [r["id"] for r in scores]
    topics_v2 = load_jsonl(Path(config.gap_list_v2_dir) / "topics.jsonl")
    cuts = {
        float(q): v
        for q, v in json.loads((Path(config.gap_list_v2_dir) / "decision.json").read_text())["cuts_by_unit_bin"].items()
    }
    pct = np.array([r["length_pct"] for r in scores])
    topic = np.array([r["topic"] for r in scores])
    keys = [template_key(p, config.prefix_chars) for p in prompts]
    topic_score = {r["topic"]: r["score"] for r in topics_v2}
    topic_flag = {r["topic"]: {q: r[f"flag_{q}"] for q in QS} for r in topics_v2}
    t_scores = np.array(sorted(topic_score.values(), reverse=True))
    all_units = units_of(np.arange(len(pct)), keys, pct)

    def flag(score: float, n_units: int, q: float) -> bool:
        return bool(score > cuts[q][int(np.digitize(n_units, UNIT_BINS[1:-1], right=True))])

    # template share in dolci on traffic-sized subsamples, so the count rule means the same in both pools
    mix_text = pd.read_parquet(config.mix_text_path, columns=["text"])["text"].tolist()
    sample = np.sort(np.random.default_rng(config.seed).choice(len(mix_text), size=config.n_mix, replace=False))
    assert [t["mix_index"] for t in d_tags] == sample.tolist()
    dolci_texts = [mix_text[i] for i in sample]
    del mix_text
    templ_sub = float(
        np.mean(
            [
                np.mean(template_flags([dolci_texts[i] for i in rng.choice(config.n_mix, len(pct), replace=False)]))
                for _ in range(config.n_template_subsamples)
            ]
        )
    )

    by_topic = {int(t): np.where(topic == t)[0] for t in np.unique(topic)}
    kinds, cells = {}, []
    for k in ALL_KINDS:
        is_k = np.array([t[k] is True for t in tags])
        idx = np.where(is_k)[0]
        u = units_of(idx, keys, pct)
        med = float(np.median(u))
        rand = np.array(
            [np.median(rng.choice(all_units, size=len(u), replace=False)) for _ in range(config.n_random_groups)]
        )
        cut_ranks = [
            int((t_scores > np.median(rng.choice(u, size=min(config.cut_units, len(u)), replace=False))).sum()) + 1
            for _ in range(config.n_cuts)
        ]
        diffs = []
        for t, ti in by_topic.items():
            a, b = units_of(ti[is_k[ti]], keys, pct), units_of(ti[~is_k[ti]], keys, pct)
            if len(a) >= config.min_units and len(b) >= config.min_units:
                diffs.append(float(np.median(a) - np.median(b)))
        diffs = np.array(diffs)
        boot = rng.choice(diffs, size=(config.n_boot, len(diffs))).mean(1) if len(diffs) else np.array([np.nan])

        def cell_count(mask: np.ndarray, record: bool) -> dict:
            n = {q: 0 for q in QS}
            for t, ti in by_topic.items():
                cu = units_of(ti[mask[ti]], keys, pct)
                if len(cu) < config.min_cell_units:
                    continue
                for q in QS:
                    if flag(float(np.median(cu)), len(cu), q) and not topic_flag[t][q]:
                        n[q] += 1
                        if record and q == 0.95:
                            cells.append(
                                {
                                    "id": f"{k}|c{t}",
                                    "kind": k,
                                    "topic": t,
                                    "n_units": len(cu),
                                    "score": round(float(np.median(cu)), 4),
                                    "topic_score": topic_score[t],
                                    "examples": [prompts[j][:200] for j in ti[mask[ti]][:3]],
                                }
                            )
            return n

        observed = cell_count(is_k, True)
        shuffled = []
        for _ in range(config.n_shuffles):
            perm = is_k.copy()
            for ti in by_topic.values():
                perm[ti] = rng.permutation(is_k[ti])
            shuffled.append(cell_count(perm, False))
        dv = np.array([t[k] for t in d_tags], dtype=object)
        kinds[k] = {
            "n_traffic": int(is_k.sum()),
            "n_units": len(u),
            "share_traffic": round(float(is_k.mean()), 4),
            "share_dolci": round(templ_sub if k == "template" else float((dv == True).mean()), 4),  # noqa: E712
            "null_traffic": int(sum(t[k] is None for t in tags)),
            "null_dolci": int(sum(v is None for v in dv)),
            "score": round(med, 4),
            "pct_among_random_groups": round(float((rand < med).mean()), 4),
            "random_group_median": round(float(np.median(rand)), 4),
            "rank_among_topics_at_21_units": int(np.median(cut_ranks)),
            "rank_at_21_units_ci90": [int(x) for x in np.percentile(cut_ranks, [5, 95])],
            "within_topic": {
                "n_topics": len(diffs),
                "mean_diff": round(float(diffs.mean()), 4) if len(diffs) else None,
                "ci95": [round(float(x), 4) for x in np.nanpercentile(boot, [2.5, 97.5])],
            },
            "cells_over_line_topic_not": {str(q): observed[q] for q in QS},
            "cells_shuffled_mean": {str(q): round(float(np.mean([s[q] for s in shuffled])), 2) for q in QS},
            "cells_shuffled_p95": {str(q): float(np.percentile([s[q] for s in shuffled], 95)) for q in QS},
            "examples": [prompts[j][:200] for j in idx[np.argsort(-pct[idx])][:3]],
        }
        LOGGER.info(
            f"{k}: {kinds[k]['score']=} {kinds[k]['pct_among_random_groups']=} {kinds[k]['rank_among_topics_at_21_units']=}"
        )
    write_jsonl(sorted(cells, key=lambda c: -c["score"]), out / "cells.jsonl")
    decision = {
        "n_traffic": len(pct),
        "n_dolci": len(d_tags),
        "dolci_template_share_full_pool": round(float(np.mean([t["template"] for t in d_tags])), 4),
        "dolci_template_share_traffic_sized": round(templ_sub, 4),
        "kinds": kinds,
    }
    write_json(decision, out / "decision.json")
    print(f"Results saved to {out / 'decision.json'} and {out / 'cells.jsonl'}")
    print(
        json.dumps(
            {
                k: {
                    x: v[x]
                    for x in (
                        "share_traffic",
                        "share_dolci",
                        "score",
                        "pct_among_random_groups",
                        "rank_among_topics_at_21_units",
                        "within_topic",
                        "cells_over_line_topic_not",
                        "cells_shuffled_mean",
                    )
                }
                for k, v in kinds.items()
            },
            indent=1,
        )
    )


if __name__ == "__main__":
    main()
