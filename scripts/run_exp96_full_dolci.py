# exp96 follow-up: each sampled traffic prompt's nearest prompt in the full dolci mix (minilm), used as a near-copy filter on the olmo differential ranking and as a training-only score on its own, scored against the user's topic labels.
# uv run --no-sync -m scripts.run_exp96_full_dolci --exp96_dir results/raw/261029_exp96_traffic_ranking_v1 --exp95_dir results/raw/261028_exp95_group_removal_v1 --exp37_dir results/raw/261021_exp37_gap_pipeline_v2 --labels_path results/raw/261029_exp96_labels_v1/decision.json --output_dir results/raw/261029_exp96_full_dolci_v1 --seed 0

import json
import logging
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import simple_parsing
import torch
from sklearn.metrics import roc_auc_score

from src.data.gap_pool import load_mix, load_traffic_pool
from src.data.io import load_jsonl
from src.data.pools import normalise
from src.metrics.gap_pipeline import text_hash
from src.metrics.io import write_json, write_jsonl
from src.utils.config import Config, write_config_json
from src.utils.logging import setup_logging
from src.utils.seed import set_seed
from scripts.run_exp96_traffic_ranking import Exp96Config, load_labels

LOGGER = logging.getLogger(__name__)


@dataclass
class FullDolciConfig(Config):
    exp96_dir: str = ""
    exp95_dir: str = ""
    exp37_dir: str = ""
    labels_path: str = ""
    copy_cos: float = 0.95
    copy_shares: list[float] = field(default_factory=lambda: [0.3, 0.5, 0.7])
    main_share: float = 0.5
    n_boot: int = 10000
    seed: int = 0


def auroc(y: np.ndarray, s: np.ndarray) -> float:
    return round(float(roc_auc_score(y, s)), 4)


def main() -> None:
    config = simple_parsing.parse(FullDolciConfig, add_config_path_arg=True)
    out = Path(config.output_dir)
    setup_logging(out)
    write_config_json(config, out)
    set_seed(config.seed)

    traffic = [r for r in load_jsonl(Path(config.exp95_dir) / "prompts.jsonl") if r["kind"] == "traffic"]
    c37 = json.loads((Path(config.exp37_dir) / "config.json").read_text())
    rows, t_np, _, _ = load_traffic_pool(
        c37["exp09_dir"], c37["exp31_dir"], c37["emb_v2"], c37["new_lmsys_shards"], c37["embed_chars"], device="cpu"
    )
    pool_idx = np.array([int(r["id"][1:]) for r in traffic])
    assert all(
        rows[i]["text"] == r["prompt"] for i, r in zip(pool_idx[:500], traffic[:500])
    ), "ids do not index the pool"
    m_np, m_hash, m_text = load_mix(c37["sft_path"], c37["emb_v1"], with_text=True)

    # nearest dolci prompt for every sampled traffic prompt, in chunks on the gpu
    m_t = torch.from_numpy(m_np).cuda()
    q = torch.from_numpy(t_np[pool_idx].astype(np.float16)).cuda()
    best_cos, best_j = [], []
    for start in range(0, len(q), 2048):
        v, j = (q[start : start + 2048] @ m_t.T).float().max(dim=1)
        best_cos.append(v.cpu().numpy())
        best_j.append(j.cpu().numpy())
    best_cos, best_j = np.concatenate(best_cos), np.concatenate(best_j)
    del m_t
    exact = np.isin([text_hash(normalise(r["prompt"])) for r in traffic], m_hash)
    LOGGER.info(f"{exact.mean()=:.4f} {(best_cos >= config.copy_cos).mean()=:.4f}")
    write_jsonl(
        [
            {"id": r["id"], "topic": r["topic"], "nearest_cos": round(float(c), 4), "exact_in_dolci": bool(e)}
            for r, c, e in zip(traffic, best_cos, exact)
        ],
        out / "prompt_nearest.jsonl",
    )

    topic = np.array([r["topic"] for r in traffic])
    topics = sorted(set(topic.tolist()))
    members = {t: np.where(topic == t)[0] for t in topics}
    is_copy = (best_cos >= config.copy_cos) | exact
    copy_share = np.array([is_copy[members[t]].mean() for t in topics])
    # training-only score on its own: distance to the nearest of all 1.7m dolci prompts, topic median
    full_dist = np.array([np.median(1 - best_cos[members[t]]) for t in topics])

    ts = {int(r["id"][1:]): r for r in load_jsonl(Path(config.exp96_dir) / "topic_scores.jsonl")}
    olmo_diff = np.array([ts[t]["olmo_diff"] for t in topics])
    tfidf_diff = np.array([ts[t]["tfidf_diff"] for t in topics])
    length = np.array([ts[t]["length"] for t in topics])

    old = load_labels(Exp96Config())
    new = {int(r["id"][1:]): r["label"] for r in json.loads(Path(config.labels_path).read_text())["rows"]}
    label_sets = {"old_117": old, "old_plus_new": {**old, **new}}

    results = {}
    for name, human in label_sets.items():
        idx = np.array([j for j, t in enumerate(topics) if human.get(t) in ("real_gap", "covered")])
        y = np.array([human[topics[j]] == "real_gap" for j in idx], dtype=int)
        res = {"n_real": int(y.sum()), "n_covered": int(len(y) - y.sum())}
        for s_name, s in (
            ("olmo_diff", olmo_diff),
            ("tfidf_diff", tfidf_diff),
            ("full_dolci_dist", full_dist),
            ("length", length),
        ):
            res[s_name] = auroc(y, s[idx])
        # the filter: topics with copy share at or above the cut drop to the bottom of the ranking
        for share in config.copy_shares:
            kept = copy_share[idx] < share
            res[f"filter_{share}"] = {
                "dropped_real": int((~kept & (y == 1)).sum()),
                "dropped_covered": int((~kept & (y == 0)).sum()),
                "olmo_diff_filtered": auroc(y, np.where(copy_share[idx] < share, olmo_diff[idx], -1e9)),
                "olmo_diff_on_kept_only": auroc(y[kept], olmo_diff[idx][kept]),
            }
        results[name] = res

    kept_all = copy_share < config.main_share
    order = [j for j in np.argsort(-olmo_diff) if kept_all[j]][:30]
    human = label_sets["old_plus_new"]
    top30 = [
        {
            "rank": r + 1,
            "topic": int(topics[j]),
            "olmo_diff": round(float(olmo_diff[j]), 4),
            "copy_share": round(float(copy_share[j]), 3),
            "label": human.get(topics[j]),
            "example": traffic[members[topics[j]][0]]["prompt"][:200],
        }
        for r, j in enumerate(order)
    ]
    write_jsonl(
        [
            {
                "id": f"c{t}",
                "copy_share": round(float(copy_share[j]), 4),
                "full_dolci_dist": round(float(full_dist[j]), 4),
                "olmo_diff": round(float(olmo_diff[j]), 4),
                "label": human.get(t),
            }
            for j, t in enumerate(topics)
        ],
        out / "topic_scores.jsonl",
    )
    write_json(top30, out / "top30_filtered.json")
    decision = {
        "n_prompts": len(traffic),
        "share_exact_in_dolci": round(float(exact.mean()), 4),
        "share_near_copy": round(float(is_copy.mean()), 4),
        "topics_dropped": {str(s): int((copy_share >= s).sum()) for s in config.copy_shares},
        "results": results,
        "top30_filtered_labels": dict(Counter(r["label"] for r in top30)),
    }
    write_json(decision, out / "decision.json")
    print(f"Results saved to {out / 'decision.json'}")
    print(json.dumps(decision, indent=1))


if __name__ == "__main__":
    main()
