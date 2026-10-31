# within_cluster_source: tests whether delta-knn separates traffic from dolci chat once the pooled cluster is held fixed.
# uv run --no-sync -m scripts.run_within_cluster_source --gap_list_v2_dir results/raw/261030_gap_list_v2 --pooled_dir results/raw/261030_pooled_chat_clusters_v1 --chat_path data/processed/chat_kinds/chat_kinds.jsonl --output_dir results/raw/261031_within_cluster_source_v1 --model_id allenai/Olmo-3-7B-Instruct-DPO --seed 0

import os

os.environ.setdefault("OMP_NUM_THREADS", "16")

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import simple_parsing
import torch
from sklearn.metrics import roc_auc_score

from scripts.run_pooled_chat_clusters import kth_distances
from src.data.io import load_jsonl
from src.interp.train_only_detectors import knn_distances
from src.metrics.io import write_json
from src.utils.config import Config, write_config_json
from src.utils.logging import setup_logging
from src.utils.seed import set_seed

LOGGER = logging.getLogger(__name__)


@dataclass
class WithinClusterConfig(Config):
    model_id: str = "allenai/Olmo-3-7B-Instruct-DPO"
    gap_list_v2_dir: str = ""
    pooled_dir: str = ""
    chat_path: str = ""
    mix_text_path: str = "cache/safety_removal/mix_text.parquet"
    feats_path: str = "/dev/shm/gap_list_v2_feats.npy"
    k: int = 100
    dup_extra: int = 50
    dup_eps: float = 0.05
    min_each: int = 3
    n_boot: int = 2000
    chunk: int = 2048
    seed: int = 0


def main() -> None:
    config = simple_parsing.parse(WithinClusterConfig)
    out = Path(config.output_dir)
    out.mkdir(parents=True, exist_ok=False)
    setup_logging(out)
    set_seed(config.seed)
    write_config_json(config, out)
    rng = np.random.default_rng(config.seed)

    scores = load_jsonl(Path(config.gap_list_v2_dir) / "prompt_scores.jsonl")
    rows = np.load(Path(config.gap_list_v2_dir) / "rows.npz")
    n_t, n_mix, n_ref, _ = rows["sizes"].tolist()
    # dolci chat rows and their scores, exactly as pooled_chat_clusters and fair_chat_line
    n_mix_text = len(pd.read_parquet(config.mix_text_path, columns=["text"]))
    sample = np.sort(np.random.default_rng(config.seed).choice(n_mix_text, size=n_mix, replace=False))
    pos = {int(m): i for i, m in enumerate(sample)}
    chat = np.sort([pos[r["mix_index"]] for r in load_jsonl(config.chat_path)])
    x = torch.tensor(np.load(config.feats_path), device="cuda").float()
    xmix, xdolma = x[n_t : n_t + n_mix], x[n_t + n_mix : n_t + n_mix + n_ref]
    sd = xmix.std(0) + 1e-6
    xc = torch.tensor(chat, device="cuda")
    c_plain, _ = kth_distances(xmix / sd, (xmix / sd)[xc], config.k + 1, config.dup_extra, config.dup_eps, config.chunk)
    dolma_c = knn_distances(xdolma, xmix[xc], (config.k,))[config.k].cpu().numpy()
    del x, xmix, xdolma
    torch.cuda.empty_cache()
    score = np.r_[np.array([r["delta_knn"] for r in scores]), c_plain - dolma_c]
    is_traffic = np.r_[np.ones(n_t, dtype=bool), np.zeros(len(chat), dtype=bool)]
    lab = np.load(Path(config.pooled_dir) / "pooled_cluster_labels.npy")
    assert len(lab) == len(score)
    LOGGER.info(f"{len(score)=} {np.median(score[is_traffic])=:.4f} {np.median(score[~is_traffic])=:.4f}")

    frame = pd.DataFrame({"g": lab, "s": score, "t": is_traffic})
    counts = frame.groupby("g")["t"].agg(["sum", "size"])
    counts["dolci"] = counts["size"] - counts["sum"]
    mixed = counts.index[(counts["sum"] >= config.min_each) & (counts["dolci"] >= config.min_each)].to_numpy()
    m = frame[frame["g"].isin(mixed)].copy()
    m["c"] = m["s"] - m.groupby("g")["s"].transform("median")
    per = m.groupby("g").apply(lambda d: roc_auc_score(d["t"], d["s"]), include_groups=False)

    by_g = {g: d for g, d in m.groupby("g")}
    boot = []
    for _ in range(config.n_boot):
        d = pd.concat([by_g[g] for g in rng.choice(mixed, len(mixed))])
        boot.append(roc_auc_score(d["t"], d["c"]))

    decision = {
        "n_pool": int(len(frame)),
        "n_clusters": int(counts.shape[0]),
        "n_mixed_clusters": int(len(mixed)),
        "n_prompts_in_mixed": int(len(m)),
        "traffic_share_in_mixed": round(float(m["t"].mean()), 4),
        "auroc_within_cluster_centred": round(float(roc_auc_score(m["t"], m["c"])), 4),
        "auroc_within_cluster_centred_ci95": [round(float(v), 4) for v in np.percentile(boot, [2.5, 97.5])],
        "auroc_raw_same_prompts": round(float(roc_auc_score(m["t"], m["s"])), 4),
        "auroc_raw_whole_pool": round(float(roc_auc_score(frame["t"], frame["s"])), 4),
        "per_cluster_auroc_median": round(float(per.median()), 4),
        "per_cluster_auroc_iqr": [round(float(v), 4) for v in per.quantile([0.25, 0.75])],
        "score_medians": {
            "traffic": round(float(np.median(score[is_traffic])), 4),
            "dolci_chat": round(float(np.median(score[~is_traffic])), 4),
        },
    }
    write_json(decision, out / "decision.json")
    print(f"Results saved to {out / 'decision.json'}")
    print(json.dumps(decision, indent=1))


if __name__ == "__main__":
    main()
