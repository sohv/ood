# exp132: clusters all traffic prompts in olmo layer-20 space into as many clusters as topics and scores them exactly like topics.
# uv run --no-sync -m scripts.run_exp132_olmo_clusters_v2 --exp130_dir results/raw/261030_exp130_gap_list_v2 --tags_path data/processed/exp116/traffic_tags.jsonl --output_dir results/raw/261030_exp132_olmo_clusters_v2 --model_id anthropic/claude-sonnet-4.6 --seed 0

import asyncio
import json
import logging
import os
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import simple_parsing
import torch
from openai import AsyncOpenAI

from scripts.run_exp114_gap_kinds import OPEN_PROMPT, messages_block, parse_fields
from scripts.run_exp130_gap_list_v2 import QS, UNIT_BINS, template_key
from src.data.io import load_jsonl
from src.generation.prompt_kinds import ALL_KINDS, NAME_TOKEN
from src.interp.kmeans import spherical_kmeans
from src.metrics.discovery import name_clusters
from src.metrics.io import write_json, write_jsonl
from src.utils.config import Config, write_config_json
from src.utils.logging import setup_logging
from src.utils.seed import set_seed

LOGGER = logging.getLogger(__name__)
FIELDS = ["NAME", "DESCRIPTION", "DIMENSION", "SHARED"]


@dataclass
class Exp132Config(Config):
    model_id: str = "anthropic/claude-sonnet-4.6"
    exp130_dir: str = ""
    exp112_dir: str = "results/raw/261029_exp112_gap_list_v1"
    tags_path: str = ""
    feats_path: str = "/dev/shm/exp130_feats.npy"
    cache_dir: str = "cache/exp132"
    n_init: int = 3
    prefix_chars: int = 40
    n_label: int = 10
    n_top: int = 100
    n_top_label: int = 30
    show_chars: int = 1500
    max_concurrent: int = 16
    seed: int = 0


def contingency(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    _, ai = np.unique(a, return_inverse=True)
    _, bi = np.unique(b, return_inverse=True)
    table = np.zeros((ai.max() + 1, bi.max() + 1))
    np.add.at(table, (ai, bi), 1)
    return table


def ari(a: np.ndarray, b: np.ndarray) -> float:
    t = contingency(a, b)

    def comb(x: np.ndarray) -> float:
        return float((x * (x - 1) / 2).sum())

    total, rows, cols = comb(np.array([t.sum()])), comb(t.sum(1)), comb(t.sum(0))
    expected = rows * cols / total
    return (comb(t) - expected) / ((rows + cols) / 2 - expected)


def nmi(a: np.ndarray, b: np.ndarray) -> float:
    p = contingency(a, b) / len(a)
    pa, pb = p.sum(1), p.sum(0)
    nz = p > 0
    mi = float((p[nz] * np.log(p[nz] / np.outer(pa, pb)[nz])).sum())

    def h(x: np.ndarray) -> float:
        return float(-(x[x > 0] * np.log(x[x > 0])).sum())

    return mi / np.sqrt(h(pa) * h(pb))


def main() -> None:
    config = simple_parsing.parse(Exp132Config)
    out = Path(config.output_dir)
    out.mkdir(parents=True, exist_ok=False)
    setup_logging(out)
    set_seed(config.seed)
    write_config_json(config, out)
    torch.backends.cudnn.deterministic = True
    rng = np.random.default_rng(config.seed)

    scores = load_jsonl(Path(config.exp130_dir) / "prompt_scores.jsonl")
    prompts = [r["prompt"] for r in load_jsonl(Path(config.exp112_dir) / "prompt_scores.jsonl")]
    tags = load_jsonl(config.tags_path)
    assert [t["id"] for t in tags] == [r["id"] for r in scores]
    topics_v2 = load_jsonl(Path(config.exp130_dir) / "topics.jsonl")
    cuts = {
        float(q): v
        for q, v in json.loads((Path(config.exp130_dir) / "decision.json").read_text())["cuts_by_unit_bin"].items()
    }
    rows = np.load(Path(config.exp130_dir) / "rows.npz")
    n_t, n_mix = int(rows["sizes"][0]), int(rows["sizes"][1])
    pct = np.array([r["length_pct"] for r in scores])
    n_tok = np.array([r["n_tokens"] for r in scores])
    topic = np.array([r["topic"] for r in scores])
    keys = [template_key(p, config.prefix_chars) for p in prompts]
    has_name = np.array([bool(NAME_TOKEN.search(p)) for p in prompts])
    kind = {k: np.array([t[k] is True for t in tags]) for k in ALL_KINDS}
    n_clusters = len(topics_v2)

    feats = np.load(config.feats_path, mmap_mode="r")
    x = torch.tensor(np.asarray(feats[: n_t + n_mix]), device="cuda").float()
    xt = x[:n_t] / (x[n_t:].std(0) + 1e-6)
    del x
    labels, _ = spherical_kmeans(xt, n_clusters, n_init=config.n_init, seed=config.seed)
    cl = labels.cpu().numpy()
    np.save(out / "cluster_labels.npy", cl)
    del xt
    torch.cuda.empty_cache()

    def describe(idx: np.ndarray, name: str, gid: int) -> dict:
        units = defaultdict(list)
        for i in idx:
            units[keys[i]].append(pct[i])
        u = np.array([np.median(v) for v in units.values()])
        s = float(np.median(u))
        ub = int(np.digitize(len(u), UNIT_BINS[1:-1], right=True))
        tc = Counter(topic[idx].tolist())
        return {
            "id": f"{name}{gid}",
            "grouping": name,
            "group": gid,
            "n_prompts": len(idx),
            "n_units": len(u),
            "score": round(s, 4),
            **{f"flag_{q}": bool(s > cuts[q][ub]) for q in QS},
            "main_topic_share": round(tc.most_common(1)[0][1] / len(idx), 4),
            "n_topics": len(tc),
            "median_tokens": float(np.median(n_tok[idx])),
            "name_share": round(float(has_name[idx].mean()), 4),
            "template_share": round(1 - len(u) / len(idx), 4),
            "kind_shares": {k: round(float(kind[k][idx].mean()), 4) for k in ALL_KINDS},
            "examples": [prompts[j][:200] for j in idx[np.argsort(-pct[idx])][:3]],
        }

    groups = [describe(np.where(cl == c)[0], "k", c) for c in range(n_clusters) if (cl == c).any()]
    groups += [describe(np.where(topic == t)[0], "c", int(t)) for t in np.unique(topic)]
    ranked = {g: sorted([r for r in groups if r["grouping"] == g], key=lambda r: -r["score"]) for g in ("k", "c")}
    for g, rs in ranked.items():
        for i, r in enumerate(rs, start=1):
            r["rank"] = i

    # label the top of each grouping with the same open prompt
    jobs = []
    for g in ("k", "c"):
        for r in ranked[g][: config.n_top_label]:
            idx = np.where((cl if g == "k" else topic) == r["group"])[0]
            show = rng.choice(idx, size=min(config.n_label, len(idx)), replace=False)
            jobs.append((r, [prompts[j] for j in show]))
    client = AsyncOpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.environ["OPENROUTER_API_KEY"])
    replies = asyncio.run(
        name_clusters(
            client,
            config.model_id,
            [OPEN_PROMPT.format(n=len(t), messages=messages_block(t, config.show_chars)) for _, t in jobs],
            config.cache_dir,
            config.max_concurrent,
        )
    )
    for (r, _), reply in zip(jobs, replies, strict=True):
        f = parse_fields(reply, FIELDS)
        r["open_name"], r["open_dimension"] = f["name"], f["dimension"]
    write_jsonl(ranked["k"] + ranked["c"], out / "groups.jsonl")

    def top_stats(rs: list[dict]) -> dict:
        return {
            "median_size": float(np.median([r["n_prompts"] for r in rs])),
            "median_tokens": float(np.median([r["median_tokens"] for r in rs])),
            "name_share": round(float(np.mean([r["name_share"] for r in rs])), 4),
            "template_share": round(float(np.mean([r["template_share"] for r in rs])), 4),
            "n_mostly_one_topic": sum(r["main_topic_share"] >= 0.5 for r in rs),
            "kind_shares": {k: round(float(np.mean([r["kind_shares"][k] for r in rs])), 4) for k in ALL_KINDS},
            "n_flagged": {str(q): sum(r[f"flag_{q}"] for r in rs) for q in QS},
        }

    labelled = [r for r, _ in jobs]
    top_prompts = {
        g: set(
            np.where(np.isin(cl if g == "k" else topic, [r["group"] for r in ranked[g][: config.n_top]]))[0].tolist()
        )
        for g in ("k", "c")
    }
    decision = {
        "n_traffic": n_t,
        "n_clusters": sum(r["grouping"] == "k" for r in groups),
        "n_topics": sum(r["grouping"] == "c" for r in groups),
        "ari_clusters_topics": round(ari(cl, topic), 4),
        "nmi_clusters_topics": round(nmi(cl, topic), 4),
        "top100_clusters": top_stats(ranked["k"][: config.n_top]),
        "top100_topics": top_stats(ranked["c"][: config.n_top]),
        "all_clusters": top_stats(ranked["k"]),
        "all_topics": top_stats(ranked["c"]),
        "top100_prompt_overlap": {
            "clusters": len(top_prompts["k"]),
            "topics": len(top_prompts["c"]),
            "shared": len(top_prompts["k"] & top_prompts["c"]),
        },
        "n_null_labels": sum(r.get("open_dimension") is None for r in labelled),
        "label_dimensions": {
            g: dict(Counter(r["open_dimension"] for r in labelled if r["grouping"] == g)) for g in ("k", "c")
        },
        "top30": {
            g: [
                {
                    x: r[x]
                    for x in (
                        "rank",
                        "group",
                        "n_prompts",
                        "score",
                        "main_topic_share",
                        "median_tokens",
                        "open_name",
                        "open_dimension",
                    )
                }
                for r in ranked[g][: config.n_top_label]
            ]
            for g in ("k", "c")
        },
    }
    write_json(decision, out / "decision.json")
    print(f"Results saved to {out / 'groups.jsonl'} and {out / 'decision.json'}")
    print(json.dumps({k: v for k, v in decision.items() if k != "top30"}, indent=1))


if __name__ == "__main__":
    main()
