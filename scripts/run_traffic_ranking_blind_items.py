# traffic_ranking blind round: 10 unlabelled topics from the top of the filtered olmo differential and 10 from the top of distance to the full dolci mix, each prompt shown beside its nearest dolci prompt by tf-idf, which neither score uses.
# uv run --no-sync -m scripts.run_traffic_ranking_blind_items --run_dir results/raw/261029_traffic_ranking_blind_items_v1 --full_dolci_dir results/raw/261029_traffic_ranking_full_dolci_v1 --group_removal_dir results/raw/261028_group_removal_v1 --pipeline_million_dir results/raw/261021_pipeline_million_gap_pipeline_v2 --items_dir data/processed/human_labels_traffic_ranking_blind --seed 0

import json
import logging
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import simple_parsing
from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer

from src.data.gap_pool import load_mix
from src.data.io import load_jsonl
from src.metrics.io import write_json
from src.utils.git import get_git_hash
from src.utils.logging import setup_logging
from src.utils.seed import set_seed

LOGGER = logging.getLogger(__name__)
PLACEHOLDER = re.compile(r"NAME_\d+")


@dataclass
class BlindConfig:
    run_dir: str = ""
    full_dolci_dir: str = ""
    group_removal_dir: str = ""
    pipeline_million_dir: str = ""
    items_dir: str = ""
    seed: int = 0
    n_per_arm: int = 10
    copy_share_cut: float = 0.5
    n_shown: int = 5
    chars: int = 600
    tfidf_chars: int = 1000
    prefix_chars: int = 60
    twin_share: float = 0.3


def main():
    config = simple_parsing.parse(BlindConfig)
    run = Path(config.run_dir)
    assert not run.exists(), f"{run} exists; results/raw is append-only"
    setup_logging(run)
    set_seed(config.seed)
    write_json({"git_hash": get_git_hash(), **asdict(config)}, run / "config.json")
    rng = np.random.default_rng(config.seed)

    ts = load_jsonl(Path(config.full_dolci_dir) / "topic_scores.jsonl")
    by_topic: dict[int, list[str]] = {}
    for r in load_jsonl(Path(config.group_removal_dir) / "prompts.jsonl"):
        if r["kind"] == "traffic":
            by_topic.setdefault(r["topic"], []).append(r["prompt"])

    def prefixes(t: int) -> set[str]:
        counts: dict[str, int] = {}
        for p in by_topic[t]:
            k = " ".join(p.split())[: config.prefix_chars].lower()
            counts[k] = counts.get(k, 0) + 1
        return {p for p, n in counts.items() if n >= config.twin_share * len(by_topic[t])}

    # template twins of labelled or already picked topics are skipped, so one template is not shown twice
    seen = set().union(*(prefixes(int(r["id"][1:])) for r in ts if r["label"] is not None))
    unlabelled = [r for r in ts if r["label"] is None]

    def take(rows: list[dict], key: str, exclude: set[int]) -> list[int]:
        picked = []
        for r in sorted(rows, key=lambda r: -r[key]):
            t = int(r["id"][1:])
            pre = prefixes(t)
            if t in exclude or pre & seen:
                continue
            seen.update(pre)
            picked.append(t)
            if len(picked) == config.n_per_arm:
                break
        return picked

    olmo_arm = take([r for r in unlabelled if r["copy_share"] < config.copy_share_cut], "olmo_diff", set())
    dist_arm = take(unlabelled, "full_dolci_dist", set(olmo_arm))
    topics = olmo_arm + dist_arm
    LOGGER.info(f"{olmo_arm=} {dist_arm=}")

    c37 = json.loads((Path(config.pipeline_million_dir) / "config.json").read_text())
    _, _, m_text = load_mix(c37["sft_path"], c37["emb_v1"], with_text=True)
    # hashed tf-idf bounds memory on 1.7m texts; name placeholders are stripped so they do not dominate the match
    hv = HashingVectorizer(
        preprocessor=lambda t: PLACEHOLDER.sub(" ", t).lower(),
        ngram_range=(1, 2),
        n_features=2**20,
        alternate_sign=False,
        norm=None,
    )
    tf = TfidfTransformer(sublinear_tf=True)
    m_x = tf.fit_transform(hv.transform([t[: config.tfidf_chars] for t in m_text]))

    order = rng.permutation(len(topics))
    items = []
    for pos, k in enumerate(order):
        c = topics[k]
        shown = rng.choice(len(by_topic[c]), size=config.n_shown, replace=False)
        texts = [by_topic[c][i] for i in shown]
        q = tf.transform(hv.transform([t[: config.tfidf_chars] for t in texts]))
        best = np.asarray((q @ m_x.T).argmax(axis=1)).ravel()
        items.append(
            {
                "id": f"f96b_c{c}",
                "kind": "topic",
                "active": True,
                "order": int(pos),
                "meta": "topic",
                "examples": [
                    {"text": t[: config.chars], "nearest": m_text[j][: config.chars]}
                    for t, j in zip(texts, best, strict=True)
                ],
            }
        )
    Path(config.items_dir).mkdir(parents=True, exist_ok=True)
    write_json(items, Path(config.items_dir) / "items.json")
    write_json({"olmo_diff_filtered": olmo_arm, "full_dolci_dist": dist_arm, "topics": topics}, run / "key.json")
    print(f"Saved {len(items)} items to {Path(config.items_dir) / 'items.json'}; key in {run / 'key.json'}")


if __name__ == "__main__":
    main()
