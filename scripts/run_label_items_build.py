# builds the reply and topic items for the human-labeling page, blind to the judge verdict and the loop arms.
# uv run -m scripts.run_label_items_build --exp37_run results/raw/261021_exp37_gap_pipeline_v2 --exp37_pools data/processed/exp37_v2 --topics_path results/raw/261021_exp39_gap_loop_v2/reviewed_topics.jsonl --output_dir data/processed/human_labels --seed 0
import json
import logging
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import simple_parsing
import torch
from sentence_transformers import SentenceTransformer

from src.data.pools import load_dolci_dpo

LOGGER = logging.getLogger(__name__)


@dataclass
class LabelItemsConfig:
    exp37_run: str = ""
    exp37_pools: str = ""
    topics_path: str = ""
    emb_v1: str = "results/raw/261021_exp31_self_nature_cells_v1"
    sft_path: str = "data/processed/exp31/dolci_sft_first_user.parquet"
    embedder_id: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    embedder_revision: str = "e8f8c211226b894fcb81acc59f3b34ba3efd5f42"
    output_dir: str = ""
    n_orange: int = 60
    n_clean: int = 40
    seed: int = 0


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open()]


def main() -> None:
    config = simple_parsing.parse(LabelItemsConfig)
    logging.basicConfig(level=logging.INFO)
    run = Path(config.exp37_run)
    prompts = {p["id"]: p["text"] for p in read_jsonl(Path(config.exp37_pools) / "prompts.jsonl")}
    replies = {r["id"]: r for r in read_jsonl(run / "gen/rollouts.jsonl")}
    grades = read_jsonl(run / "gen/grades_v3.jsonl")
    # one reply per prompt so no prompt is labelled twice
    first = [g for g in grades if g["replicate"] == 0 and g["orange"] is not None]
    rng = random.Random(config.seed)
    orange = rng.sample([g for g in first if g["orange"]], config.n_orange)
    clean = rng.sample([g for g in first if not g["orange"]], config.n_clean)
    picked = orange + clean
    rng.shuffle(picked)
    items, key = [], []
    for i, g in enumerate(picked):
        items.append(
            {
                "id": f"r{i:03d}",
                "kind": "reply",
                "order": i,
                "meta": "reply",
                "prompt": prompts[g["prompt_id"]][:6000],
                "reply": replies[g["id"]]["reply"][:6000],
            }
        )
        key.append({"id": f"r{i:03d}", "grade_id": g["id"], "judge_orange": g["orange"], "scores": g["scores"]})
    out = Path(config.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "reply_items.json").write_text(json.dumps(items, indent=2, ensure_ascii=False))
    (out / "reply_key.json").write_text(json.dumps(key, indent=2))
    LOGGER.info(f"{len(items)} reply items, {config.n_orange} judged orange")
    print(f"Saved {out / 'reply_items.json'} and {out / 'reply_key.json'}")
    if config.topics_path:
        build_topics(config, out)


def build_topics(config: LabelItemsConfig, out: Path) -> None:
    topics = read_jsonl(Path(config.topics_path))
    sft = pd.read_parquet(config.sft_path, columns=["first_user"])
    sft = sft[sft["first_user"].notna()].reset_index(drop=True)
    mix_text = sft["first_user"].tolist() + load_dolci_dpo()["first_user"].tolist()
    mix = torch.cat(
        [
            torch.from_numpy(np.load(Path(config.emb_v1) / f"emb_{n}.npy", mmap_mode="r")[:].astype(np.float16)).cuda()
            for n in ("sft", "dpo")
        ]
    )
    assert len(mix) == len(mix_text), (len(mix), len(mix_text))
    embedder = SentenceTransformer(config.embedder_id, revision=config.embedder_revision, device="cuda")
    texts = [e for t in topics for e in t["examples"]]
    q = torch.from_numpy(embedder.encode(texts, normalize_embeddings=True, batch_size=256)).cuda().half()
    best = [(q[s : s + 256] @ mix.T).max(1) for s in range(0, len(q), 256)]
    sim, idx = torch.cat([b.values for b in best]), torch.cat([b.indices for b in best])
    near = iter(zip(idx.tolist(), sim.tolist(), strict=True))
    rng = random.Random(config.seed)
    order = list(range(len(topics)))
    rng.shuffle(order)
    items = []
    for pos, t in zip(order, topics, strict=True):
        examples = []
        for e in t["examples"]:
            j, s = next(near)
            examples.append({"text": e, "nearest": f"[similarity {s:.2f}] " + mix_text[j][:400]})
        items.append(
            {
                "id": t["id"],
                "kind": "topic",
                "order": pos,
                "meta": f"topic, {t['size']} prompts",
                "title": f"{t['name']}: {t['description']}",
                "examples": examples,
            }
        )
    (out / "topic_items.json").write_text(json.dumps(items, indent=2, ensure_ascii=False))
    LOGGER.info(f"{len(items)} topic items")
    print(f"Saved {out / 'topic_items.json'}")


if __name__ == "__main__":
    main()
