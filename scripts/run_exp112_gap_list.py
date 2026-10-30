# exp112: ranks real-traffic topics by delta-knn (layer 20, k 100) against exp107's 200k dolci sample, reusing exp108's cached features.
# uv run --no-sync -m scripts.run_exp112_gap_list --dataset_path results/raw/261029_exp98_clean_v1/clean_prompts.jsonl --exp111_dir results/raw/261029_exp111_chat_removal_loop_v1 --output_dir results/raw/261029_exp112_gap_list_v1 --model_id allenai/Olmo-3-7B-Instruct-DPO --seed 0

import json
import logging
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import simple_parsing
import torch

from scripts.run_exp107_traffic_removal import KINDS
from src.data.io import load_jsonl
from src.interp.train_only_detectors import knn_distances
from src.metrics.io import write_json, write_jsonl
from src.utils.config import Config, write_config_json
from src.utils.logging import setup_logging
from src.utils.seed import set_seed

LOGGER = logging.getLogger(__name__)


@dataclass
class Exp112Config(Config):
    model_id: str = "allenai/Olmo-3-7B-Instruct-DPO"
    model_revision: str = "b33130b7"
    exp111_dir: str = ""
    feats_cache: str = "/dev/shm/exp108_feats.npy"
    n_mix: int = 200000
    n_ref: int = 3000
    min_prompts: int = 5
    layer: int = 20
    k: int = 100
    n_examples: int = 5
    seed: int = 0


def main() -> None:
    config = simple_parsing.parse(Exp112Config, add_config_path_arg=True)
    out = Path(config.output_dir)
    setup_logging(out)
    write_config_json(config, out)
    set_seed(config.seed)

    # traffic filtered exactly as in exp107/exp108, so it lines up with the cached features
    traffic = load_jsonl(config.dataset_path)
    counts = Counter(r["topic"] for r in traffic)
    traffic = [r for r in traffic if counts[r["topic"]] >= config.min_prompts]
    topic = np.array([r["topic"] for r in traffic])
    topics = np.array(sorted(set(topic.tolist())))
    n_t = len(traffic)

    feats = np.load(config.feats_cache)
    assert len(feats) == n_t + config.n_mix + config.n_ref, f"{len(feats)=} {n_t=}"
    x = torch.tensor(feats, device="cuda").float()
    del feats
    xt, xmix, xdolma = x[:n_t], x[n_t : n_t + config.n_mix], x[n_t + config.n_mix :]
    to_dolma = knn_distances(xdolma, xt, (config.k,))[config.k].cpu().numpy()
    sd = xmix.std(0) + 1e-6
    to_dolci = knn_distances(xmix / sd, xt / sd, (config.k,), scale=False)[config.k].cpu().numpy()
    score = to_dolci - to_dolma

    labels = {}
    for kind, (_, _, d, fields) in KINDS.items():
        rows = {int(r["id"][1:]): any(r[f] for f in fields) for r in load_jsonl(Path(d) / "topic_labels.jsonl")}
        labels[kind] = rows

    write_jsonl(
        [
            {
                "id": r["id"],
                "topic": int(r["topic"]),
                "delta_knn": round(float(s), 4),
                "to_dolci": round(float(a), 4),
                "to_dolma": round(float(b), 4),
                "prompt": r["prompt"],
            }
            for r, s, a, b in zip(traffic, score, to_dolci, to_dolma, strict=True)
        ],
        out / "prompt_scores.jsonl",
    )
    med = np.array([np.median(score[topic == t]) for t in topics])
    order = np.argsort(-med)
    ranked = []
    for rank, i in enumerate(order):
        t = int(topics[i])
        idx = np.where(topic == t)[0]
        # examples are the topic's highest-scoring prompts
        top_idx = idx[np.argsort(-score[idx])][: config.n_examples]
        ranked.append(
            {
                "id": f"c{t}",
                "rank": rank + 1,
                "topic": t,
                "n_prompts": len(idx),
                "median_delta_knn": round(float(med[i]), 4),
                **{f"is_{k}": labels[k].get(t) for k in KINDS},
                "examples": [traffic[j]["prompt"][:300] for j in top_idx],
            }
        )
    write_jsonl(ranked, out / "topics.jsonl")

    # check: exp111's full arm, delta-knn, round 0 used the same scores
    r0 = [
        r["topics"]
        for r in load_jsonl(Path(config.exp111_dir) / "rounds.jsonl")
        if r["detector"] == "delta_knn" and r["arm"] == "full" and r["round"] == 0
    ][0]
    top20 = [r["topic"] for r in ranked[:20]]
    decision = {
        "n_prompts": n_t,
        "n_topics": len(topics),
        "n_nan": int(np.isnan(score).sum()),
        "top20_matches_exp111": top20 == r0,
        "kind_share_top": {
            k: {n: round(float(np.mean([bool(r[f"is_{k}"]) for r in ranked[:n]])), 4) for n in (20, 50, 100)}
            for k in KINDS
        },
        "top20": [{"rank": r["rank"], "topic": r["topic"], "example": r["examples"][0][:150]} for r in ranked[:20]],
    }
    write_json(decision, out / "decision.json")
    print(f"Results saved to {out / 'topics.jsonl'} and {out / 'decision.json'}")
    print(json.dumps({k: v for k, v in decision.items() if k != "top20"}, indent=1))


if __name__ == "__main__":
    main()
