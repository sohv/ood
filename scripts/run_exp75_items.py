# exp75: draws blind topic-label items from each exp70 loop arm's confirmed gaps plus random topics, each prompt shown beside its nearest training prompt.
# uv run -m scripts.run_exp75_items --run_dir results/raw/261027_exp75_items_v1 --exp70_dir results/raw/261027_exp70_loop_rankings_v1 --exp37_dir results/raw/261021_exp37_gap_pipeline_v2 --label_dirs results/raw/261022_exp41_topic_labels_v1/labels results/raw/261022_exp43_topic_labels_v1/labels results/raw/261026_exp61_labels_v1/labels --items_dir data/processed/human_labels_exp75 --seed 0

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import simple_parsing
import torch

from src.data.gap_pool import load_mix, load_traffic_pool
from src.data.io import load_jsonl
from src.metrics.io import write_json
from src.utils.git import get_git_hash
from src.utils.logging import setup_logging
from src.utils.seed import set_seed

LOGGER = logging.getLogger(__name__)


@dataclass
class Exp75Config:
    run_dir: str = ""
    exp70_dir: str = ""
    exp37_dir: str = ""
    label_dirs: list[str] = field(default_factory=list)
    items_dir: str = ""
    seed: int = 0
    n_per_arm: dict = field(default_factory=lambda: {"B": 12, "C": 12, "A": 8, "random": 8})
    n_shown: int = 10
    chars: int = 600


def main():
    config = simple_parsing.parse(Exp75Config)
    assert config.run_dir and config.exp70_dir and config.exp37_dir and config.items_dir, "paths required"
    run = Path(config.run_dir)
    assert not run.exists(), f"{run} exists; results/raw is append-only"
    setup_logging(run)
    set_seed(config.seed)
    write_json({"git_hash": get_git_hash(), **asdict(config)}, run / "config.json")
    rng = np.random.default_rng(config.seed)

    labelled = {int(p.stem.split("_c")[1]) for d in config.label_dirs for p in Path(d).glob("*_c*.json")}
    conf = {
        a: [
            r["topic"]
            for r in load_jsonl(Path(config.exp70_dir) / f"reviewed_topics_{a}.jsonl")
            if r["confirmed"] and r["topic"] not in labelled
        ]
        for a in ("A", "B", "C")
    }
    clusters = load_jsonl(Path(config.exp37_dir) / "clusters.jsonl")
    ranked = [int(c["id"][1:]) for c in clusters if c["rank"] and int(c["id"][1:]) not in labelled]
    pools = {**conf, "random": ranked}
    drawn = {a: sorted(rng.choice(pools[a], size=config.n_per_arm[a], replace=False).tolist()) for a in pools}
    topics = sorted({c for v in drawn.values() for c in v})
    LOGGER.info(f"{len(labelled)} topics already labelled; drawn {drawn}; {len(topics)} distinct topics")

    c37 = json.loads((Path(config.exp37_dir) / "config.json").read_text())
    rows, t_np, _, _ = load_traffic_pool(
        c37["exp09_dir"], c37["exp31_dir"], c37["emb_v2"], c37["new_lmsys_shards"], c37["embed_chars"], device="cpu"
    )
    labels = np.load(Path(config.exp37_dir) / "labels.npy")
    m_np, _, m_text = load_mix(c37["sft_path"], c37["emb_v1"], with_text=True)
    m_t = torch.from_numpy(m_np.astype(np.float32))

    order = rng.permutation(len(topics))
    items = []
    for pos, k in enumerate(order):
        c = topics[k]
        members = np.where(labels == c)[0]
        shown = rng.choice(members, size=min(config.n_shown, len(members)), replace=False)
        best = (torch.from_numpy(t_np[shown].astype(np.float32)) @ m_t.T).argmax(1).numpy()
        items.append(
            {
                "id": f"f75t_c{c}",
                "kind": "topic",
                "active": True,
                "order": int(pos),
                "meta": "topic",
                "examples": [
                    {"text": rows[i]["text"][: config.chars], "nearest": m_text[j][: config.chars]}
                    for i, j in zip(shown, best, strict=True)
                ],
            }
        )
    Path(config.items_dir).mkdir(parents=True, exist_ok=True)
    write_json(items, Path(config.items_dir) / "items.json")
    write_json({"drawn": drawn, "topics": topics}, run / "key.json")
    print(f"Saved {len(items)} items to {Path(config.items_dir) / 'items.json'}; key in {run / 'key.json'}")


if __name__ == "__main__":
    main()
