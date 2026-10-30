# precision_at_depth: builds the hand-labelling items: 20 topics drawn by rank bin with 10 prompts each beside their nearest mix prompt, plus the 58 earlier topics for a relabel, names hidden.
# uv run -m scripts.run_build_label_items_precision_at_depth --run_dir results/raw/261026_precision_at_depth_labels_v1 --output_dir data/processed/human_labels_precision_at_depth --pipeline_million_dir results/raw/261021_pipeline_million_gap_pipeline_v2 --label_files results/raw/261022_deep_gap_loop_topic_labels_v1/decision.json results/raw/261022_classifier_gap_loop_topic_labels_v1/decision.json --item_files data/processed/human_labels_deep_gap_loop/topic_items.json data/processed/human_labels_classifier_gap_loop/topic_items.json --seed 0

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
BINS = ((1, 50), (51, 100), (101, 200), (201, 293))


@dataclass
class LabelItemsConfig:
    run_dir: str = ""
    output_dir: str = ""
    pipeline_million_dir: str = ""
    label_files: list[str] = field(default_factory=list)
    item_files: list[str] = field(default_factory=list)
    seed: int = 0
    per_bin: int = 5
    n_shown: int = 10
    max_chars: int = 400


def main():
    config = simple_parsing.parse(LabelItemsConfig)
    assert (
        config.run_dir
        and config.output_dir
        and config.pipeline_million_dir
        and config.label_files
        and config.item_files
    )
    run, out = Path(config.run_dir), Path(config.output_dir)
    assert not run.exists(), f"{run} exists; results/raw is append-only"
    setup_logging(run)
    set_seed(config.seed)
    write_json({"git_hash": get_git_hash(), **asdict(config)}, run / "config.json")
    rng = np.random.default_rng(config.seed)

    human = {}
    for path in config.label_files:
        for r in json.loads(Path(path).read_text())["rows"]:
            human[int(r["id"][1:])] = r["human"]
    c37 = json.loads((Path(config.pipeline_million_dir) / "config.json").read_text())
    clusters = {int(c["id"][1:]): c for c in load_jsonl(Path(config.pipeline_million_dir) / "clusters.jsonl")}
    by_rank = {clusters[c]["rank"]: c for c in clusters if clusters[c]["rank"]}
    new = {}
    for lo, hi in BINS:
        pool = [by_rank[r] for r in range(lo, hi + 1) if r in by_rank and by_rank[r] not in human]
        for c in rng.choice(pool, size=config.per_bin, replace=False).tolist():
            new[c] = f"{lo}-{hi}"
    LOGGER.info(f"new topics {new}")

    rows, t_np, _, _ = load_traffic_pool(
        c37["iterative_discovery_dir"],
        c37["self_nature_a_dir"],
        c37["emb_v2"],
        c37["new_lmsys_shards"],
        c37["embed_chars"],
    )
    labels = np.load(Path(config.pipeline_million_dir) / "labels.npy")
    m_np, _, m_text = load_mix(c37["sft_path"], c37["emb_v1"], with_text=True)
    m_gpu = torch.from_numpy(m_np).cuda()
    items = []
    for c in new:
        members = np.where(labels == c)[0]
        shown = np.random.default_rng(config.seed).choice(
            members, size=min(config.n_shown, len(members)), replace=False
        )
        q = torch.from_numpy(t_np[shown]).cuda()
        sims, idx = (q @ m_gpu.T).float().max(1)
        items.append(
            {
                "id": f"e61n_c{c}",
                "kind": "topic",
                "active": True,
                "meta": f"topic, {len(members)} prompts",
                "examples": [
                    {
                        "text": rows[int(i)]["text"][: config.max_chars],
                        "nearest": f"[similarity {float(s):.2f}] {m_text[int(j)][: config.max_chars]}",
                    }
                    for i, s, j in zip(shown, sims.cpu().numpy(), idx.cpu().numpy())
                ],
            }
        )
    seen = set()
    for path in config.item_files:
        for it in json.loads(Path(path).read_text()):
            c = int(it["id"].split("_c")[1])
            if c in seen:
                continue
            seen.add(c)
            # the relabel shows exactly what the first round showed, minus the topic name
            items.append(
                {"id": f"e61r_c{c}", "kind": "topic", "active": True, "meta": it["meta"], "examples": it["examples"]}
            )
    assert len(seen) == len(human) == 58, (len(seen), len(human))
    order = rng.permutation(len(items))
    for pos, i in enumerate(order):
        items[int(i)]["order"] = int(pos)
    items.sort(key=lambda x: x["order"])
    out.mkdir(parents=True, exist_ok=True)
    write_json(items, out / "topic_items.json")
    key = {
        "new": {
            f"c{c}": {"bin": b, "rank": clusters[c]["rank"], "ratio": clusters[c]["ratio"]} for c, b in new.items()
        },
        "relabel": {f"c{c}": {"first_label": human[c], "rank": clusters[c]["rank"]} for c in seen},
    }
    write_json(key, run / "key.json")
    print(f"{len(items)} items written to {out / 'topic_items.json'}; key at {run / 'key.json'}")


if __name__ == "__main__":
    main()
