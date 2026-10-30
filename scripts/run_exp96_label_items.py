# exp96 labels: blind topic-label items from the top of exp96's olmo differential ranking (template twins skipped) plus random topics, each prompt shown beside its nearest training prompt.
# uv run --no-sync -m scripts.run_exp96_label_items --run_dir results/raw/261029_exp96_label_items_v1 --exp96_dir results/raw/261029_exp96_traffic_ranking_v1 --exp37_dir results/raw/261021_exp37_gap_pipeline_v2 --items_dir data/processed/human_labels_exp96 --seed 0

import json
import logging
from collections import Counter
from dataclasses import asdict, dataclass
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
class Exp96LabelConfig:
    run_dir: str = ""
    exp96_dir: str = ""
    exp37_dir: str = ""
    items_dir: str = ""
    seed: int = 0
    n_top: int = 20
    n_random: int = 5
    n_shown: int = 10
    chars: int = 600
    prefix_chars: int = 60
    twin_share: float = 0.3


def main():
    config = simple_parsing.parse(Exp96LabelConfig)
    assert config.run_dir and config.exp96_dir and config.exp37_dir and config.items_dir, "paths required"
    run = Path(config.run_dir)
    assert not run.exists(), f"{run} exists; results/raw is append-only"
    setup_logging(run)
    set_seed(config.seed)
    write_json({"git_hash": get_git_hash(), **asdict(config)}, run / "config.json")
    rng = np.random.default_rng(config.seed)

    topic_rows = load_jsonl(Path(config.exp96_dir) / "topic_scores.jsonl")
    sample = load_jsonl(Path(config.exp96_dir).parent / "261028_exp95_group_removal_v1" / "prompts.jsonl")
    by_topic: dict[int, list[str]] = {}
    for r in sample:
        if r["kind"] == "traffic":
            by_topic.setdefault(r["topic"], []).append(r["prompt"])

    def prefixes(t: int) -> set[str]:
        # prefixes shared by at least twin_share of a topic's 30 sampled prompts mark a template
        counts = Counter(" ".join(p.split())[: config.prefix_chars].lower() for p in by_topic[t])
        return {p for p, n in counts.items() if n >= config.twin_share * len(by_topic[t])}

    unlabelled = [r for r in topic_rows if r["label"] is None]
    labelled_prefixes = set().union(*(prefixes(int(r["id"][1:])) for r in topic_rows if r["label"] is not None))
    top, seen, skipped = [], set(labelled_prefixes), []
    for r in sorted(unlabelled, key=lambda r: -r["olmo_diff"]):
        t = int(r["id"][1:])
        pre = prefixes(t)
        if pre & seen:
            skipped.append(t)
            continue
        seen |= pre
        top.append(t)
        if len(top) == config.n_top:
            break
    rest = [int(r["id"][1:]) for r in unlabelled if int(r["id"][1:]) not in set(top) | set(skipped)]
    rand = sorted(rng.choice(rest, size=config.n_random, replace=False).tolist())
    topics = sorted(top + rand)
    LOGGER.info(f"{top=} {skipped=} {rand=}")

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
                "id": f"f96t_c{c}",
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
    write_json({"top": top, "skipped_twins": skipped, "random": rand, "topics": topics}, run / "key.json")
    print(f"Saved {len(items)} items to {Path(config.items_dir) / 'items.json'}; key in {run / 'key.json'}")


if __name__ == "__main__":
    main()
