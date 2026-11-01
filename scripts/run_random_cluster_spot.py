# random_cluster_spot: builds a blind label sample of 20 random olmo clusters mixed with 10 already-labelled top groups.
# uv run --no-sync -m scripts.run_random_cluster_spot --gap_list_v2_dir results/raw/261030_gap_list_v2 --olmo_clusters_v2_dir results/raw/261030_olmo_clusters_v2 --cluster_gap_list_dir results/raw/261030_cluster_gap_list_v2 --spot_dir results/raw/261030_cluster_gap_list_v1 --labels_path data/processed/human_labels_cluster_gap_list/labels.json --output_dir results/raw/261031_random_cluster_spot_v1 --model_id allenai/Olmo-3-7B-Instruct-DPO --seed 0

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import simple_parsing

from src.data.io import load_jsonl
from src.metrics.io import write_json, write_jsonl
from src.utils.config import Config, write_config_json
from src.utils.logging import setup_logging
from src.utils.seed import set_seed

LOGGER = logging.getLogger(__name__)


@dataclass
class RandomSpotConfig(Config):
    model_id: str = "allenai/Olmo-3-7B-Instruct-DPO"
    gap_list_v2_dir: str = ""
    gap_list_dir: str = "results/raw/261029_gap_list_v1"
    olmo_clusters_v2_dir: str = ""
    cluster_gap_list_dir: str = ""
    spot_dir: str = ""
    labels_path: str = ""
    feats_path: str = "/dev/shm/gap_list_v2_feats.npy"
    min_cluster: int = 10
    n_random: int = 20
    n_relabel_each: int = 5
    top_n: int = 20
    n_examples: int = 5
    show_chars: int = 1200
    seed: int = 0


def main() -> None:
    config = simple_parsing.parse(RandomSpotConfig)
    out = Path(config.output_dir)
    out.mkdir(parents=True, exist_ok=False)
    setup_logging(out)
    set_seed(config.seed)
    write_config_json(config, out)
    rng = np.random.default_rng(config.seed)

    prompts = [r["prompt"] for r in load_jsonl(Path(config.gap_list_dir) / "prompt_scores.jsonl")]
    n_t = len(prompts)
    # same float32 merge of small clusters as cluster_gap_list v2
    cl = np.load(Path(config.olmo_clusters_v2_dir) / "cluster_labels.npy")
    assert len(cl) == n_t
    feats = np.load(config.feats_path, mmap_mode="r")
    n_mix = int(np.load(Path(config.gap_list_v2_dir) / "rows.npz")["sizes"][1])
    sd = feats[n_t : n_t + n_mix].astype(np.float32).std(0) + 1e-6
    xt = feats[:n_t].astype(np.float32) / sd
    sizes = np.bincount(cl)
    big = np.where(sizes >= config.min_cluster)[0]
    cent = np.stack([xt[cl == c].mean(0) for c in big])
    cent /= np.linalg.norm(cent, axis=1, keepdims=True)
    small = np.where(sizes[cl] < config.min_cluster)[0]
    clusters = cl.copy()
    clusters[small] = big[(xt[small] / np.linalg.norm(xt[small], axis=1, keepdims=True) @ cent.T).argmax(1)]
    del xt

    groups = [
        g for g in load_jsonl(Path(config.cluster_gap_list_dir) / "groups.jsonl") if g["grouping"] == "olmo_clusters"
    ]
    assert len(groups) == len(np.unique(clusters)), (len(groups), len(np.unique(clusters)))
    pool = [g for g in groups if g["rank"] > config.top_n]
    picks = rng.choice(len(pool), size=config.n_random, replace=False)
    items = []
    for i in picks:
        g = pool[i]
        idx = np.where(clusters == g["group"])[0]
        assert len(idx) == g["n_prompts"], (g["group"], len(idx), g["n_prompts"])
        ex = rng.choice(idx, size=min(config.n_examples, len(idx)), replace=False)
        items.append(
            {
                "kind": "random_cluster",
                "group": int(g["group"]),
                "rank": int(g["rank"]),
                "examples": [prompts[j][: config.show_chars] for j in ex],
            }
        )

    # relabel arm: already-labelled top groups, shown with the exact prompts the user saw
    old = {json.loads(line)["id"]: json.loads(line) for line in open(Path(config.spot_dir) / "spot_items.jsonl")}
    labels = json.load(open(config.labels_path))
    for source in ("olmo_clusters", "topics"):
        cand = [l for l in labels if l["source"] == source]
        for k in rng.choice(len(cand), size=config.n_relabel_each, replace=False):
            l = cand[k]
            items.append(
                {
                    "kind": f"relabel_{source}",
                    "group": int(l["group"]),
                    "rank": int(l["rank"]),
                    "old_id": l["id"],
                    "old_verdict": l["verdict"],
                    "examples": old[l["id"]]["examples"],
                }
            )

    order = rng.permutation(len(items))
    page, key = [], {}
    for pos, j in enumerate(order):
        rid = f"r{pos:02d}"
        key[rid] = {k: v for k, v in items[j].items() if k != "examples"}
        page.append(
            {"id": rid, "order": pos, "n_examples": len(items[j]["examples"]), "examples": items[j]["examples"]}
        )
    write_jsonl(page, out / "spot_items.jsonl")
    write_json(key, out / "spot_key.json")
    LOGGER.info(f"{len(page)=} random ranks {sorted(items[i]['rank'] for i in range(config.n_random))}")
    print(f"Results saved to {out / 'spot_items.jsonl'} and {out / 'spot_key.json'}")


if __name__ == "__main__":
    main()
