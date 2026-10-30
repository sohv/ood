# messy_controls: asks whether dolci already covers pasted real-world text with a task, and whether dolci's own real-chat rows (messy but covered) score like the traffic gaps.
# uv run --no-sync -m scripts.run_messy_controls --gap_list_v2_dir results/raw/261030_gap_list_v2 --olmo_clusters_v2_dir results/raw/261030_olmo_clusters_v2 --cluster_gap_list_dir results/raw/261030_cluster_gap_list_v1 --tags_dir data/processed/kind_groups --chat_path data/processed/chat_kinds/chat_kinds.jsonl --output_dir results/raw/261030_messy_controls_v1 --model_id allenai/Olmo-3-7B-Instruct-DPO --seed 0

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import simple_parsing
import torch
from sklearn.metrics import roc_auc_score

from src.data.io import load_jsonl
from src.interp.train_only_detectors import knn_distances
from src.metrics.io import write_json, write_jsonl
from src.utils.config import Config, write_config_json
from src.utils.logging import setup_logging
from src.utils.seed import set_seed

LOGGER = logging.getLogger(__name__)
TASK = re.compile(r"summar|rewrite|re-write|rephrase|paraphras|extract|condense|shorten|simplif", re.IGNORECASE)


@dataclass
class MessyConfig(Config):
    model_id: str = "allenai/Olmo-3-7B-Instruct-DPO"
    gap_list_v2_dir: str = ""
    gap_list_dir: str = "results/raw/261029_gap_list_v1"
    olmo_clusters_v2_dir: str = ""
    cluster_gap_list_dir: str = ""
    tags_dir: str = ""
    chat_path: str = ""
    mix_text_path: str = "cache/safety_removal/mix_text.parquet"
    feats_path: str = "/dev/shm/gap_list_v2_feats.npy"
    k: int = 100
    min_cluster: int = 10
    n_top_clusters: int = 50
    n_other: int = 30000
    n_neighbour_clusters: int = 10
    n_neighbour_prompts: int = 3
    n_neighbours: int = 10
    show_chars: int = 400
    seed: int = 0


def summarise(v: np.ndarray) -> dict:
    return {
        "n": len(v),
        "median": round(float(np.median(v)), 4),
        "p25": round(float(np.percentile(v, 25)), 4),
        "p75": round(float(np.percentile(v, 75)), 4),
    }


def main() -> None:
    config = simple_parsing.parse(MessyConfig)
    out = Path(config.output_dir)
    out.mkdir(parents=True, exist_ok=False)
    setup_logging(out)
    set_seed(config.seed)
    write_config_json(config, out)
    rng = np.random.default_rng(config.seed)

    scores = load_jsonl(Path(config.gap_list_v2_dir) / "prompt_scores.jsonl")
    prompts = [r["prompt"] for r in load_jsonl(Path(config.gap_list_dir) / "prompt_scores.jsonl")]
    t_tags = load_jsonl(Path(config.tags_dir) / "traffic_tags.jsonl")
    d_tags = load_jsonl(Path(config.tags_dir) / "dolci_tags.jsonl")
    assert [t["id"] for t in t_tags] == [r["id"] for r in scores]
    rows = np.load(Path(config.gap_list_v2_dir) / "rows.npz")
    n_t, n_mix, n_ref, _ = rows["sizes"].tolist()

    mix_text = pd.read_parquet(config.mix_text_path, columns=["text"])["text"].tolist()
    sample = np.sort(np.random.default_rng(config.seed).choice(len(mix_text), size=n_mix, replace=False))
    assert [t["mix_index"] for t in d_tags] == sample.tolist()
    dolci_texts = [mix_text[i] for i in sample]
    del mix_text
    pos = {int(m): i for i, m in enumerate(sample)}
    is_chat = np.zeros(n_mix, dtype=bool)
    is_chat[[pos[r["mix_index"]] for r in load_jsonl(config.chat_path)]] = True

    feats = np.load(config.feats_path)
    x = torch.tensor(feats, device="cuda").float()
    del feats
    xt, xmix, xdolma = x[:n_t], x[n_t : n_t + n_mix], x[n_t + n_mix : n_t + n_mix + n_ref]
    sd = xmix.std(0) + 1e-6
    xt_s, xmix_s = xt / sd, xmix / sd

    # traffic scored as in gap_list_v2; dolci rows leave themselves out by taking the (k+1)-th neighbour
    s_t = (
        (
            knn_distances(xmix_s, xt_s, (config.k,), scale=False)[config.k]
            - knn_distances(xdolma, xt, (config.k,))[config.k]
        )
        .cpu()
        .numpy()
    )
    saved = np.array([r["delta_knn"] for r in scores])
    LOGGER.info(f"recomputed vs saved traffic score: {np.corrcoef(s_t, saved)[0, 1]=} {np.abs(s_t - saved).max()=}")
    other = rng.choice(np.where(~is_chat)[0], size=config.n_other, replace=False)
    d_idx = np.concatenate([np.where(is_chat)[0], other])
    xd = torch.tensor(d_idx, device="cuda")
    s_d = (
        (
            knn_distances(xmix_s, xmix_s[xd], (config.k + 1,), scale=False)[config.k + 1]
            - knn_distances(xdolma, xmix[xd], (config.k,))[config.k]
        )
        .cpu()
        .numpy()
    )
    s_chat, s_other = s_d[: is_chat.sum()], s_d[is_chat.sum() :]

    # traffic prompts in the top olmo clusters of cluster_gap_list, clusters rebuilt the same way
    cl = np.load(Path(config.olmo_clusters_v2_dir) / "cluster_labels.npy")
    xt_np = xt_s.cpu().numpy()
    sizes = np.bincount(cl)
    big = np.where(sizes >= config.min_cluster)[0]
    cent = np.stack([xt_np[cl == c].mean(0) for c in big])
    cent /= np.linalg.norm(cent, axis=1, keepdims=True)
    small = np.where(sizes[cl] < config.min_cluster)[0]
    clusters = cl.copy()
    clusters[small] = big[(xt_np[small] / np.linalg.norm(xt_np[small], axis=1, keepdims=True) @ cent.T).argmax(1)]
    groups = [
        r for r in load_jsonl(Path(config.cluster_gap_list_dir) / "groups.jsonl") if r["grouping"] == "olmo_clusters"
    ]
    top_ids = [r["group"] for r in groups[: config.n_top_clusters]]
    in_top = np.isin(clusters, top_ids)

    # pasted real-world text with a task, from the kind tags plus a task-word match on the first 300 characters
    def pasted_task(texts: list[str], tags: list[dict]) -> np.ndarray:
        return np.array(
            [t["pasted_document"] is True and bool(TASK.search(x[:300])) for x, t in zip(texts, tags, strict=True)]
        )

    t_pt = pasted_task(prompts, t_tags)
    d_pt = pasted_task(dolci_texts, d_tags)
    t_pd = np.array([t["pasted_document"] is True for t in t_tags])
    d_pd = np.array([t["pasted_document"] is True for t in d_tags])
    d_pt_idx = d_pt[d_idx]

    coverage = {
        "pasted_document_share": {
            "traffic": round(float(t_pd.mean()), 4),
            "dolci_sample": round(float(d_pd.mean()), 4),
            "dolci_chat_rows": round(float(d_pd[is_chat].mean()), 4),
            "traffic_top_clusters": round(float(t_pd[in_top].mean()), 4),
        },
        "pasted_text_with_task_share": {
            "traffic": round(float(t_pt.mean()), 4),
            "dolci_sample": round(float(d_pt.mean()), 4),
            "dolci_chat_rows": round(float(d_pt[is_chat].mean()), 4),
            "traffic_top_clusters": round(float(t_pt[in_top].mean()), 4),
        },
        "pasted_text_with_task_count": {"traffic": int(t_pt.sum()), "dolci_sample": int(d_pt.sum())},
        "null_tags_dolci": int(sum(t["pasted_document"] is None for t in d_tags)),
        "null_tags_traffic": int(sum(t["pasted_document"] is None for t in t_tags)),
    }

    gap = s_t[in_top]
    messy = {
        "score_traffic_all": summarise(s_t),
        "score_traffic_top_clusters": summarise(gap),
        "score_dolci_chat_rows_leave_one_out": summarise(s_chat),
        "score_dolci_other_rows_leave_one_out": summarise(s_other),
        "score_traffic_pasted_task": summarise(s_t[t_pt]),
        "score_dolci_pasted_task_leave_one_out": summarise(s_d[d_pt_idx]) if d_pt_idx.any() else None,
        "auroc_top_clusters_vs_dolci_chat": round(
            float(roc_auc_score(np.r_[np.ones(len(gap)), np.zeros(len(s_chat))], np.r_[gap, s_chat])), 4
        ),
        "auroc_traffic_vs_dolci_chat": round(
            float(roc_auc_score(np.r_[np.ones(n_t), np.zeros(len(s_chat))], np.r_[s_t, s_chat])), 4
        ),
        "auroc_dolci_chat_vs_dolci_other": round(
            float(roc_auc_score(np.r_[np.ones(len(s_chat)), np.zeros(len(s_other))], np.r_[s_chat, s_other])), 4
        ),
        "auroc_traffic_pasted_task_vs_dolci_pasted_task": round(
            float(roc_auc_score(np.r_[np.ones(t_pt.sum()), np.zeros(d_pt_idx.sum())], np.r_[s_t[t_pt], s_d[d_pt_idx]])),
            4,
        )
        if d_pt_idx.any()
        else None,
        "share_dolci_chat_above_top_cluster_median": round(float((s_chat > np.median(gap)).mean()), 4),
        "share_traffic_above_top_cluster_median": round(float((s_t > np.median(gap)).mean()), 4),
    }

    # nearest dolci prompts for random prompts from the top clusters, to read what dolci has closest
    xmix_n = torch.nn.functional.normalize(xmix_s, dim=1)
    neighbour_rows, has_pasted = [], []
    for g in top_ids[: config.n_neighbour_clusters]:
        idx = np.where(clusters == g)[0]
        for j in rng.choice(idx, size=min(config.n_neighbour_prompts, len(idx)), replace=False):
            sims, nn = (torch.nn.functional.normalize(xt_s[j : j + 1], dim=1) @ xmix_n.T).topk(config.n_neighbours)
            nn = nn[0].cpu().numpy()
            has_pasted.append(bool(d_pd[nn].any()))
            neighbour_rows.append(
                {
                    "id": f"c{g}|t{j}",
                    "cluster": int(g),
                    "traffic_prompt": prompts[j][: config.show_chars],
                    "neighbours": [
                        {
                            "distance": round(float(np.sqrt(max(0.0, 2 - 2 * float(s)))), 4),
                            "pasted_document": d_tags[int(n)]["pasted_document"],
                            "chat_row": bool(is_chat[n]),
                            "text": dolci_texts[int(n)][: config.show_chars],
                        }
                        for s, n in zip(sims[0].cpu().numpy(), nn, strict=True)
                    ],
                }
            )
    coverage["share_top_cluster_prompts_with_pasted_dolci_neighbour_in_10"] = round(float(np.mean(has_pasted)), 4)
    write_jsonl(neighbour_rows, out / "neighbours.jsonl")

    decision = {
        "n_traffic": n_t,
        "n_dolci_chat_rows": int(is_chat.sum()),
        "n_dolci_other_scored": config.n_other,
        "n_top_cluster_prompts": int(in_top.sum()),
        "coverage": coverage,
        "messy_control": messy,
    }
    write_json(decision, out / "decision.json")
    print(f"Results saved to {out / 'decision.json'} and {out / 'neighbours.jsonl'}")
    print(json.dumps(decision, indent=1))


if __name__ == "__main__":
    main()
