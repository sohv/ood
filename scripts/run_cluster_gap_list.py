# cluster_gap_list: ranks real traffic gaps by raw delta-knn under olmo clusters (min 10 prompts) and under topics, with size-matched random-group lines, quirk rates, and a blind spot-check sample.
# uv run --no-sync -m scripts.run_cluster_gap_list --gap_list_v2_dir results/raw/261030_gap_list_v2 --olmo_clusters_v2_dir results/raw/261030_olmo_clusters_v2 --tags_dir data/processed/kind_groups --output_dir results/raw/261030_cluster_gap_list_v1 --model_id allenai/Olmo-3-7B-Instruct-DPO --seed 0

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import simple_parsing

from scripts.run_gap_list_v2 import template_key
from src.data.io import load_jsonl
from src.generation.prompt_kinds import ALL_KINDS
from src.metrics.io import write_json, write_jsonl
from src.utils.config import Config, write_config_json
from src.utils.logging import setup_logging
from src.utils.seed import set_seed

LOGGER = logging.getLogger(__name__)
UNIT_BINS = (0, 10, 20, 40, 80, np.inf)
QS = (0.95, 0.99)
TOPS = (20, 50, 100)


@dataclass
class ClusterGapListConfig(Config):
    model_id: str = "allenai/Olmo-3-7B-Instruct-DPO"
    gap_list_v2_dir: str = ""
    olmo_clusters_v2_dir: str = ""
    gap_list_dir: str = "results/raw/261029_gap_list_v1"
    tags_dir: str = ""
    feats_path: str = "/dev/shm/gap_list_v2_feats.npy"
    min_cluster: int = 10
    prefix_chars: int = 40
    n_null: int = 2000
    n_spot: int = 20
    n_examples: int = 5
    show_chars: int = 1200
    seed: int = 0


def group_scores(score: np.ndarray, groups: np.ndarray, keys: list[str]) -> pd.DataFrame:
    # median over template units, then median over units in the group
    frame = pd.DataFrame({"g": groups, "key": keys, "s": score})
    units = frame.groupby(["g", "key"])["s"].median()
    return pd.DataFrame(
        {
            "score": units.groupby(level=0).median(),
            "n_units": units.groupby(level=0).size(),
            "n_prompts": frame.groupby("g").size(),
        }
    ).sort_values("score", ascending=False, kind="stable")


def main() -> None:
    config = simple_parsing.parse(ClusterGapListConfig)
    out = Path(config.output_dir)
    out.mkdir(parents=True, exist_ok=False)
    setup_logging(out)
    set_seed(config.seed)
    write_config_json(config, out)
    rng = np.random.default_rng(config.seed)

    scores = load_jsonl(Path(config.gap_list_v2_dir) / "prompt_scores.jsonl")
    prompts = [r["prompt"] for r in load_jsonl(Path(config.gap_list_dir) / "prompt_scores.jsonl")]
    tags = load_jsonl(Path(config.tags_dir) / "traffic_tags.jsonl")
    assert [t["id"] for t in tags] == [r["id"] for r in scores]
    n_t = len(scores)
    raw = np.array([r["delta_knn"] for r in scores])
    topic = np.array([r["topic"] for r in scores])
    keys = [template_key(p, config.prefix_chars) for p in prompts]

    # small olmo clusters join the nearest big cluster by centroid, as in grouping_test
    cl = np.load(Path(config.olmo_clusters_v2_dir) / "cluster_labels.npy")
    assert len(cl) == n_t
    xt = np.load(config.feats_path, mmap_mode="r")[:n_t].astype(np.float32)
    rows = np.load(Path(config.gap_list_v2_dir) / "rows.npz")
    n_mix = int(rows["sizes"][1])
    sd = np.load(config.feats_path, mmap_mode="r")[n_t : n_t + n_mix].astype(np.float32).std(0) + 1e-6
    xt = xt / sd
    sizes = np.bincount(cl)
    big = np.where(sizes >= config.min_cluster)[0]
    cent = np.stack([xt[cl == c].mean(0) for c in big])
    cent /= np.linalg.norm(cent, axis=1, keepdims=True)
    small = np.where(sizes[cl] < config.min_cluster)[0]
    clusters = cl.copy()
    clusters[small] = big[(xt[small] / np.linalg.norm(xt[small], axis=1, keepdims=True) @ cent.T).argmax(1)]
    del xt

    key_arr = np.array(keys, dtype=object)

    def random_group_score(pick: np.ndarray) -> float:
        return float(pd.Series(raw[pick]).groupby(key_arr[pick]).median().median())

    groupings = {"olmo_clusters": clusters, "topics": topic}
    tag_frame = pd.DataFrame([{k: t[k] is True for k in ALL_KINDS} for t in tags])
    quirk = (tag_frame["placeholder"] | tag_frame["template"]).to_numpy()

    tables, summary = {}, {}
    for name, groups in groupings.items():
        gs = group_scores(raw, groups, keys)
        # size-matched line: random traffic groups with the same unit count, binned
        bins = np.digitize(gs["n_units"], UNIT_BINS[1:-1], right=True)
        cuts = {}
        for b in np.unique(bins):
            size = int(np.median(gs["n_prompts"][bins == b]))
            null = [random_group_score(rng.choice(n_t, size=size, replace=False)) for _ in range(config.n_null)]
            cuts[int(b)] = {q: float(np.quantile(null, q)) for q in QS}
            LOGGER.info(f"{name} bin {b}: {size=} {cuts[int(b)]=}")
        for q in QS:
            gs[f"flag_{q}"] = [s > cuts[int(b)][q] for s, b in zip(gs["score"], bins, strict=True)]
        member = pd.Series(np.arange(n_t)).groupby(groups)
        gs["quirk_share"] = [float(quirk[member.get_group(g)].mean()) for g in gs.index]
        gs["main_topic_share"] = [
            float(pd.Series(topic[member.get_group(g)]).value_counts(normalize=True).iloc[0]) for g in gs.index
        ]
        tables[name] = gs
        summary[name] = {
            "n_groups": len(gs),
            "median_size": float(gs["n_prompts"].median()),
            "n_flagged": {str(q): int(gs[f"flag_{q}"].sum()) for q in QS},
            "prompts_in_flagged": {str(q): int(gs.loc[gs[f"flag_{q}"], "n_prompts"].sum()) for q in QS},
            "quirk_share_top": {str(n): round(float(gs["quirk_share"].iloc[:n].mean()), 4) for n in TOPS},
            "kind_shares_top50": {
                k: round(float(tag_frame[k].to_numpy()[np.isin(groups, gs.index[:50])].mean()), 4) for k in ALL_KINDS
            },
        }

    # how much of the cluster list is new relative to the topic list
    top_c = np.isin(clusters, tables["olmo_clusters"].index[:50])
    top_t = np.isin(topic, tables["topics"].index[:50])
    summary["top50_prompt_overlap"] = {
        "cluster_prompts_in_top50_topics": round(float(top_t[top_c].mean()), 4),
        "topic_prompts_in_top50_clusters": round(float(top_c[top_t].mean()), 4),
    }
    summary["quirk_share_all_traffic"] = round(float(quirk.mean()), 4)

    group_rows = []
    for name, gs in tables.items():
        groups = groupings[name]
        for rank, (g, r) in enumerate(gs.iterrows()):
            idx = np.where(groups == g)[0]
            group_rows.append(
                {
                    "id": f"{name}|{g}",
                    "grouping": name,
                    "group": int(g),
                    "rank": rank + 1,
                    "score": round(float(r["score"]), 4),
                    "n_prompts": int(r["n_prompts"]),
                    "n_units": int(r["n_units"]),
                    **{f"flag_{q}": bool(r[f"flag_{q}"]) for q in QS},
                    "quirk_share": round(float(r["quirk_share"]), 4),
                    "main_topic_share": round(float(r["main_topic_share"]), 4),
                    "examples": [prompts[j][:300] for j in idx[np.argsort(-raw[idx])][:3]],
                }
            )
    write_jsonl(group_rows, out / "groups.jsonl")

    # blind spot check: top clusters and top topics shuffled together, random examples within each group
    spot, key = [], {}
    for name, gs in tables.items():
        for g in gs.index[: config.n_spot]:
            idx = np.where(groupings[name] == g)[0]
            pick = rng.choice(idx, size=min(config.n_examples, len(idx)), replace=False)
            spot.append({"source": name, "group": int(g), "examples": [prompts[j][: config.show_chars] for j in pick]})
    order = rng.permutation(len(spot))
    spot_items = []
    for i, j in enumerate(order):
        sid = f"s{i:02d}"
        key[sid] = {"source": spot[j]["source"], "group": spot[j]["group"]}
        spot_items.append(
            {"id": sid, "order": i, "n_examples": len(spot[j]["examples"]), "examples": spot[j]["examples"]}
        )
    write_jsonl(spot_items, out / "spot_items.jsonl")
    write_json(key, out / "spot_key.json")

    write_json(summary, out / "decision.json")
    print(f"Results saved to {out / 'decision.json'}, {out / 'groups.jsonl'} and {out / 'spot_items.jsonl'}")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
