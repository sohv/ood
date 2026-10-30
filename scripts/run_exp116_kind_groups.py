# exp116: tags traffic and the 200k dolci sample with 13 prompt kinds and scores each kind group with delta-knn and exp115's line.
# uv run --no-sync -m scripts.run_exp116_kind_groups --exp112_dir results/raw/261029_exp112_gap_list_v1 --line_path results/raw/261030_exp115_gap_line_v1/line.json --tags_dir data/processed/exp116 --output_dir results/raw/261030_exp116_kind_groups_v1 --model_id google/gemini-3.8-flash --seed 0

import asyncio
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import simple_parsing
from openai import AsyncOpenAI

from src.data.io import load_jsonl
from src.generation.prompt_kinds import ALL_KINDS, LLM_KINDS, PROMPT, VALUES, VERSION, rule_tags, show
from src.generation.request_grader import grade_requests
from src.metrics.io import write_json, write_jsonl
from src.utils.config import Config, write_config_json
from src.utils.logging import setup_logging
from src.utils.seed import set_seed

LOGGER = logging.getLogger(__name__)


@dataclass
class Exp116Config(Config):
    model_id: str = "google/gemini-3.8-flash"
    exp112_dir: str = ""
    line_path: str = ""
    tags_dir: str = ""
    mix_text_path: str = "cache/exp80/mix_text.parquet"
    cache_dir: str = "cache/exp116"
    n_mix: int = 200000
    show_chars: int = 1500
    min_in_topic: int = 3
    min_subgroup: int = 5
    n_boot: int = 10000
    n_spot: int = 20
    max_concurrent: int = 48
    seed: int = 0


def tag(config: Exp116Config, client, texts: list[str], name: str) -> list[dict]:
    grades, failures = asyncio.run(
        grade_requests(
            client,
            config.model_id,
            [show(t, config.show_chars) for t in texts],
            str(Path(config.cache_dir) / name),
            config.max_concurrent,
            prompt_template=PROMPT,
            values=VALUES,
            version=VERSION,
        )
    )
    LOGGER.info(f"{name}: failures {dict(failures)}")
    rules = rule_tags(texts)
    return [
        {**r, **{k: None if g is None else g[k] == "yes" for k in LLM_KINDS}}
        for r, g in zip(rules, grades, strict=True)
    ]


def main() -> None:
    config = simple_parsing.parse(Exp116Config)
    out = Path(config.output_dir)
    out.mkdir(parents=True, exist_ok=False)
    setup_logging(out)
    set_seed(config.seed)
    write_config_json(config, out)
    tags_dir = Path(config.tags_dir)
    tags_dir.mkdir(parents=True, exist_ok=True)

    traffic = load_jsonl(Path(config.exp112_dir) / "prompt_scores.jsonl")
    # exp107's sample, drawn the same way
    mix_text = pd.read_parquet(config.mix_text_path, columns=["text"])["text"].tolist()
    sample = np.sort(np.random.default_rng(config.seed).choice(len(mix_text), size=config.n_mix, replace=False))
    if config.num_tasks:
        traffic, sample = traffic[: config.num_tasks], sample[: config.num_tasks]
    dolci_texts = [mix_text[i] for i in sample]
    del mix_text

    client = AsyncOpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.environ["OPENROUTER_API_KEY"])
    t_tags = tag(config, client, [r["prompt"] for r in traffic], "traffic")
    write_jsonl(
        [{"id": r["id"], "topic": r["topic"], **t} for r, t in zip(traffic, t_tags, strict=True)],
        tags_dir / "traffic_tags.jsonl",
    )
    d_tags = tag(config, client, dolci_texts, "dolci")
    write_jsonl(
        [{"id": f"mix{int(i)}", "mix_index": int(i), **t} for i, t in zip(sample, d_tags, strict=True)],
        tags_dir / "dolci_tags.jsonl",
    )

    cuts = {float(q): v for q, v in json.loads(Path(config.line_path).read_text())["cuts"].items()}
    score = np.array([r["delta_knn"] for r in traffic])
    topic = np.array([r["topic"] for r in traffic])
    topic_med = {t["topic"]: t["median_delta_knn"] for t in load_jsonl(Path(config.exp112_dir) / "topics.jsonl")}
    meds = np.array(sorted(topic_med.values(), reverse=True))
    flagged90 = {t for t, m in topic_med.items() if m > cuts[0.9]}
    in_flagged = np.array([t in flagged90 for t in topic])

    def crossed(m: float) -> float | None:
        return max((q for q, c in cuts.items() if m > c), default=None)

    rng = np.random.default_rng(config.seed)
    kinds, subgroups = {}, []
    for k in ALL_KINDS:
        tv = np.array([t[k] for t in t_tags], dtype=object)
        dv = np.array([t[k] for t in d_tags], dtype=object)
        is_k = tv == True  # noqa: E712
        med = float(np.median(score[is_k])) if is_k.any() else None
        # within-topic: tagged minus untagged median, for topics with enough of both
        diffs = []
        for t in np.unique(topic[is_k]):
            m_t = topic == t
            a, b = score[m_t & is_k], score[m_t & (tv == False)]  # noqa: E712
            if len(a) >= config.min_in_topic and len(b) >= config.min_in_topic:
                diffs.append(float(np.median(a) - np.median(b)))
            if len(a) >= config.min_subgroup and crossed(float(np.median(a))) and not crossed(topic_med[int(t)]):
                subgroups.append(
                    {
                        "kind": k,
                        "topic": int(t),
                        "n": int(len(a)),
                        "median": round(float(np.median(a)), 4),
                        "line": crossed(float(np.median(a))),
                        "topic_median": topic_med[int(t)],
                        "examples": [traffic[j]["prompt"][:200] for j in np.where(m_t & is_k)[0][:3]],
                    }
                )
        diffs = np.array(diffs)
        boot = rng.choice(diffs, size=(config.n_boot, len(diffs))).mean(1) if len(diffs) else np.array([np.nan])
        kinds[k] = {
            "n_traffic": int(is_k.sum()),
            "share_traffic": round(float(is_k.mean()), 4),
            "share_dolci": round(float((dv == True).mean()), 4),  # noqa: E712
            "null_traffic": int(sum(v is None for v in tv)),
            "null_dolci": int(sum(v is None for v in dv)),
            "median_delta_knn": None if med is None else round(med, 4),
            "line_crossed": None if med is None else crossed(med),
            "rank_among_topics": None if med is None else int((meds > med).sum()) + 1,
            "n_topics_spanned": int(len(np.unique(topic[is_k]))),
            "share_in_flagged90_topics": round(float(in_flagged[is_k].mean()), 4) if is_k.any() else None,
            "within_topic": {
                "n_topics": int(len(diffs)),
                "mean_diff": round(float(diffs.mean()), 4) if len(diffs) else None,
                "ci95": [round(float(x), 4) for x in np.nanpercentile(boot, [2.5, 97.5])],
            },
            "examples": [traffic[j]["prompt"][:200] for j in np.where(is_k)[0][np.argsort(-score[is_k])][:3]],
        }
        LOGGER.info(f"{k}: {kinds[k]['share_traffic']=} {kinds[k]['share_dolci']=} {kinds[k]['median_delta_knn']=}")
    write_jsonl(
        [{"id": f"{s['kind']}|c{s['topic']}", **s} for s in sorted(subgroups, key=lambda s: -s["median"])],
        out / "subgroups.jsonl",
    )

    spot = rng.choice(len(traffic), size=min(config.n_spot, len(traffic)), replace=False)
    lines = ["# exp116 spot check: are the kind tags right?\n"]
    for j in spot:
        on = [k for k in ALL_KINDS if t_tags[j][k]]
        lines.append(
            f"## {traffic[j]['id']} ({len(traffic[j]['prompt']):,} chars)\n\nTags: {', '.join(on) or 'none'}\n"
        )
        lines.append("```\n" + traffic[j]["prompt"][:1500] + "\n```\n")
    (out / "spot_check.md").write_text("\n".join(lines))

    decision = {
        "n_traffic": len(traffic),
        "n_dolci": len(sample),
        "overall_median_delta_knn": round(float(np.median(score)), 4),
        "cuts": cuts,
        "n_topics": len(topic_med),
        "n_flagged90_topics": len(flagged90),
        "n_subgroups_over_line_parent_not": len(subgroups),
        "subgroups_by_kind": {k: sum(s["kind"] == k for s in subgroups) for k in ALL_KINDS},
        "kinds": kinds,
    }
    write_json(decision, out / "decision.json")
    print(f"Results saved to {out / 'decision.json'}, {out / 'subgroups.jsonl'}, {out / 'spot_check.md'}")
    print(
        json.dumps(
            {
                k: {
                    x: v[x]
                    for x in ("share_traffic", "share_dolci", "median_delta_knn", "line_crossed", "rank_among_topics")
                }
                for k, v in kinds.items()
            },
            indent=1,
        )
    )


if __name__ == "__main__":
    main()
