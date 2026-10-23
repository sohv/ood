# exp54: builds the user's labelling items: round 1 topic-level harm for exp53's 65 targets, round 2 held-out prompt harm for the user's harmless topics.
# uv run -m scripts.run_fix2_label_items --round 1 --exp53_data data/processed/exp53 --exp53_train results/raw/261023_exp53_data_v1 --exp37_pools data/processed/exp37_v2 --exp41_dir results/raw/261022_exp41_gap_loop_v1 --output_dir data/processed/human_labels_exp54 --seed 0
import json
import logging
import random
from dataclasses import dataclass
from pathlib import Path

import simple_parsing

from src.data.io import load_jsonl

LOGGER = logging.getLogger(__name__)


@dataclass
class Fix2LabelConfig:
    round: int = 1
    exp53_data: str = ""
    exp53_train: str = ""
    exp37_pools: str = ""
    exp41_dir: str = ""
    output_dir: str = ""
    # round 2: the round-1 label files (doc id f54t_c<topic>.json, field harmless)
    round1_labels: str = ""
    per_topic: int = 5
    chars: int = 600
    min_targets: int = 10
    seed: int = 0


def clip(text: str, chars: int) -> str:
    # long prompts often share a template opening, so show the start and the end, where the actual request sits
    if len(text) <= chars:
        return text
    head = chars * 2 // 5
    return f"{text[:head]}\n[...]\n{text[-(chars - head):]}"


def main() -> None:
    config = simple_parsing.parse(Fix2LabelConfig)
    logging.basicConfig(level=logging.INFO)
    rng = random.Random(config.seed)
    out = Path(config.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    names = {int(t["id"][1:]): t["name"] for t in load_jsonl(Path(config.exp41_dir) / "reviewed_topics.jsonl")}
    train = load_jsonl(Path(config.exp53_train) / "train_gap.jsonl")
    topics = sorted({x["topic"] for x in train})
    items = []
    if config.round == 1:
        for pos, c in enumerate(rng.sample(topics, len(topics))):
            ex = rng.sample([x for x in train if x["topic"] == c], config.per_topic)
            items.append(
                {
                    "id": f"f54t_c{c}",
                    "kind": "topicharm",
                    "active": True,
                    "order": pos,
                    "meta": "topic",
                    "title": names.get(c, f"topic {c}"),
                    "examples": [{"text": clip(x["text"], config.chars)} for x in ex],
                }
            )
        (out / "round1_items.json").write_text(json.dumps(items, indent=2, ensure_ascii=False) + "\n")
    else:
        lab = {
            int(p.stem.split("_c")[1]): json.loads(p.read_text())["harmless"]
            for p in Path(config.round1_labels).glob("f54t_c*.json")
        }
        assert len(lab) == len(topics), f"{len(lab)} round-1 labels for {len(topics)} topics"
        targets = sorted(c for c, v in lab.items() if v == "yes")
        LOGGER.info(f"{len(targets)} topics labelled harmless")
        assert (
            len(targets) >= config.min_targets
        ), f"only {len(targets)} harmless topics; the pre-registration stops here"
        held = [
            t
            for t in load_jsonl(Path(config.exp53_data) / "test.jsonl")
            if t["set"] == "gap" and t["topic"] in set(targets)
        ]
        rng.shuffle(held)
        for pos, t in enumerate(held):
            items.append(
                {
                    "id": f"f54h_{t['id']}",
                    "kind": "harm",
                    "active": True,
                    "order": pos,
                    "meta": "request",
                    "prompt": t["text"][:6000],
                    "reply": "",
                }
            )
        (out / "round2_items.json").write_text(json.dumps(items, indent=2, ensure_ascii=False) + "\n")
        (out / "targets.json").write_text(
            json.dumps({"targets": targets, "round1": {str(k): v for k, v in sorted(lab.items())}}, indent=2) + "\n"
        )
    LOGGER.info(f"{len(items)} items")
    print(f"Saved {len(items)} round-{config.round} items to {out}")


if __name__ == "__main__":
    main()
