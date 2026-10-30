# olmo_clusters: clusters the 2,000 highest delta-knn traffic prompts in olmo layer-20 space and compares the clusters with the topics.
# uv run --no-sync -m scripts.run_olmo_clusters --gap_list_dir results/raw/261029_gap_list_v1 --tags_path data/processed/kind_groups/traffic_tags.jsonl --line_path results/raw/261030_gap_line_v1/line.json --output_dir results/raw/261030_olmo_clusters_v1 --model_id anthropic/claude-sonnet-4.6 --seed 0

import asyncio
import json
import logging
import os
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import simple_parsing
import torch
from openai import AsyncOpenAI

from scripts.run_gap_kinds import OPEN_PROMPT, messages_block, parse_fields
from src.data.io import load_jsonl
from src.generation.prompt_kinds import ALL_KINDS
from src.interp.kmeans import spherical_kmeans
from src.metrics.discovery import name_clusters
from src.metrics.io import write_json, write_jsonl
from src.utils.config import Config, write_config_json
from src.utils.logging import setup_logging
from src.utils.seed import set_seed

LOGGER = logging.getLogger(__name__)
FIELDS = ["NAME", "DESCRIPTION", "DIMENSION", "SHARED"]


@dataclass
class OlmoClustersConfig(Config):
    model_id: str = "anthropic/claude-sonnet-4.6"
    model_revision: str = "b33130b7"
    gap_list_dir: str = ""
    tags_path: str = ""
    line_path: str = ""
    feats_cache: str = "/dev/shm/known_gap_loop_feats.npy"
    cache_dir: str = "cache/olmo_clusters"
    layer: int = 20
    n_mix: int = 200000
    n_top: int = 2000
    n_clusters: int = 40
    n_init: int = 10
    n_label: int = 10
    show_chars: int = 1500
    max_concurrent: int = 16
    seed: int = 0


def main() -> None:
    config = simple_parsing.parse(OlmoClustersConfig)
    out = Path(config.output_dir)
    out.mkdir(parents=True, exist_ok=False)
    setup_logging(out)
    set_seed(config.seed)
    write_config_json(config, out)
    torch.backends.cudnn.deterministic = True

    traffic = load_jsonl(Path(config.gap_list_dir) / "prompt_scores.jsonl")
    tags = load_jsonl(config.tags_path)
    assert [t["id"] for t in tags] == [r["id"] for r in traffic]
    n_t = len(traffic)
    score = np.array([r["delta_knn"] for r in traffic])
    topic = np.array([r["topic"] for r in traffic])
    topic_med = {t["topic"]: t["median_delta_knn"] for t in load_jsonl(Path(config.gap_list_dir) / "topics.jsonl")}
    cuts = {float(q): v for q, v in json.loads(Path(config.line_path).read_text())["cuts"].items()}

    feats = np.load(config.feats_cache)
    x = torch.tensor(feats[: n_t + config.n_mix], device="cuda").float()
    del feats
    # the detector's own space: scaled by the dolci sample's per-dimension std
    xt = x[:n_t] / (x[n_t:].std(0) + 1e-6)
    del x

    rng = np.random.default_rng(config.seed)
    sets = {
        "top": np.argsort(-score)[: config.n_top],
        "random": rng.choice(n_t, size=config.n_top, replace=False),
    }
    rows, label_jobs = [], []
    for name, idx in sets.items():
        labels, _ = spherical_kmeans(
            xt[torch.tensor(idx, device="cuda")], config.n_clusters, config.n_init, seed=config.seed
        )
        labels = labels.cpu().numpy()
        for c in range(config.n_clusters):
            members = idx[labels == c]
            if not len(members):
                continue
            tc = Counter(topic[members].tolist())
            main_topic, main_n = tc.most_common(1)[0]
            med = float(np.median(score[members]))
            show = rng.choice(members, size=min(config.n_label, len(members)), replace=False)
            topic_rows = np.where(topic == main_topic)[0]
            show_topic = rng.choice(topic_rows, size=min(config.n_label, len(topic_rows)), replace=False)
            label_jobs += [[traffic[j]["prompt"] for j in show], [traffic[j]["prompt"] for j in show_topic]]
            rows.append(
                {
                    "id": f"{name}|k{c}",
                    "set": name,
                    "cluster": c,
                    "n": int(len(members)),
                    "median_delta_knn": round(med, 4),
                    "line_crossed": max((q for q, v in cuts.items() if med > v), default=None),
                    "main_topic": int(main_topic),
                    "main_topic_share": round(main_n / len(members), 4),
                    "main_topic_median": topic_med[int(main_topic)],
                    "n_topics": len(tc),
                    "kind_shares": {
                        k: round(float(np.mean([tags[j][k] is True for j in members])), 4) for k in ALL_KINDS
                    },
                    "examples": [traffic[j]["prompt"][:200] for j in members[np.argsort(-score[members])][:3]],
                }
            )
        LOGGER.info(f"{name}: clustered {len(idx)} prompts")

    client = AsyncOpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.environ["OPENROUTER_API_KEY"])
    prompts = [OPEN_PROMPT.format(n=len(t), messages=messages_block(t, config.show_chars)) for t in label_jobs]
    replies = asyncio.run(name_clusters(client, config.model_id, prompts, config.cache_dir, config.max_concurrent))
    for i, r in enumerate(rows):
        c, t = parse_fields(replies[2 * i], FIELDS), parse_fields(replies[2 * i + 1], FIELDS)
        r["open_name"], r["open_description"], r["open_dimension"] = c["name"], c["description"], c["dimension"]
        r["topic_open_name"], r["topic_open_dimension"] = t["name"], t["dimension"]
    write_jsonl(rows, out / "clusters.jsonl")

    def summary(sel: list[dict]) -> dict:
        dims = [r["open_dimension"] for r in sel]
        return {
            "n_clusters": len(sel),
            "n_mostly_one_topic": sum(r["main_topic_share"] >= 0.5 for r in sel),
            "mean_main_topic_share": round(float(np.mean([r["main_topic_share"] for r in sel])), 4),
            "median_topics_spanned": float(np.median([r["n_topics"] for r in sel])),
            "n_over_line": {
                q: sum(r["line_crossed"] is not None and r["line_crossed"] >= q for r in sel) for q in cuts
            },
            "open_dimensions": dict(Counter(dims)),
            "n_null_labels": sum(d is None for d in dims),
            "non_subject_share": round(
                sum(d not in (None, "subject") for d in dims) / max(1, sum(d is not None for d in dims)), 4
            ),
            "topic_label_non_subject_share": round(
                sum(r["topic_open_dimension"] not in (None, "subject") for r in sel)
                / max(1, sum(r["topic_open_dimension"] is not None for r in sel)),
                4,
            ),
            "kind_shares": {k: round(float(np.mean([r["kind_shares"][k] for r in sel])), 4) for k in ALL_KINDS},
        }

    decision = {
        "n_traffic": n_t,
        "n_rows": len(rows),
        "n_rows_expected": 2 * config.n_clusters,
        "n_label_calls": len(prompts),
        "n_null_replies": sum(r is None for r in replies),
        "top_set_score_range": [round(float(score[sets["top"]].min()), 4), round(float(score[sets["top"]].max()), 4)],
        "summary": {name: summary([r for r in rows if r["set"] == name]) for name in sets},
    }
    write_json(decision, out / "decision.json")
    print(f"Results saved to {out / 'clusters.jsonl'} and {out / 'decision.json'}")
    print(json.dumps(decision["summary"], indent=1))


if __name__ == "__main__":
    main()
