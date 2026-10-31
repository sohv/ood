# fair_chat_line: gives each grouping (olmo clusters, tf-idf clusters, topics, random) its own line set by covered dolci chat groups and counts traffic groups over it against chance.
# uv run --no-sync -m scripts.run_fair_chat_line --gap_list_v2_dir results/raw/261030_gap_list_v2 --pooled_dir results/raw/261030_pooled_chat_clusters_v1 --olmo_clusters_v2_dir results/raw/261030_olmo_clusters_v2 --tfidf_dir results/raw/261030_tfidf_cluster_gap_list_v1 --chat_path data/processed/chat_kinds/chat_kinds.jsonl --output_dir results/raw/261031_fair_chat_line_v1 --model_id allenai/Olmo-3-7B-Instruct-DPO --seed 0

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
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

from scripts.run_gap_list_v2 import template_key
from scripts.run_pooled_chat_clusters import group_scores, kth_distances
from src.data.io import load_jsonl
from src.interp.kmeans import spherical_kmeans
from src.interp.train_only_detectors import knn_distances
from src.metrics.io import write_json
from src.utils.config import Config, write_config_json
from src.utils.logging import setup_logging
from src.utils.seed import set_seed

LOGGER = logging.getLogger(__name__)
QS = (0.95, 0.99)


@dataclass
class FairLineConfig(Config):
    model_id: str = "allenai/Olmo-3-7B-Instruct-DPO"
    gap_list_v2_dir: str = ""
    gap_list_dir: str = "results/raw/261029_gap_list_v1"
    pooled_dir: str = ""
    olmo_clusters_v2_dir: str = ""
    tfidf_dir: str = ""
    chat_path: str = ""
    mix_text_path: str = "cache/safety_removal/mix_text.parquet"
    feats_path: str = "/dev/shm/gap_list_v2_feats.npy"
    encoder_id: str = "sentence-transformers/all-MiniLM-L6-v2"
    k: int = 100
    dup_extra: int = 50
    dup_eps: float = 0.05
    topic_count: int = 1903
    min_cluster: int = 10
    svd_dim: int = 256
    n_init: int = 3
    prefix_chars: int = 40
    n_boot: int = 2000
    chunk: int = 2048
    seed: int = 0


def merge_small(lab: np.ndarray, z: np.ndarray, min_cluster: int) -> np.ndarray:
    # clusters under min_cluster join the nearest bigger cluster by centroid
    sizes = np.bincount(lab)
    big = np.where(sizes >= min_cluster)[0]
    zn = z / (np.linalg.norm(z, axis=1, keepdims=True) + 1e-9)
    cent = np.stack([zn[lab == c].mean(0) for c in big])
    cent /= np.linalg.norm(cent, axis=1, keepdims=True)
    small = np.where(sizes[lab] < min_cluster)[0]
    out = lab.copy()
    out[small] = big[(zn[small] @ cent.T).argmax(1)]
    return out


def main() -> None:
    config = simple_parsing.parse(FairLineConfig)
    out = Path(config.output_dir)
    out.mkdir(parents=True, exist_ok=False)
    setup_logging(out)
    set_seed(config.seed)
    write_config_json(config, out)
    rng = np.random.default_rng(config.seed)
    torch.backends.cudnn.deterministic = True

    scores = load_jsonl(Path(config.gap_list_v2_dir) / "prompt_scores.jsonl")
    prompts = [r["prompt"] for r in load_jsonl(Path(config.gap_list_dir) / "prompt_scores.jsonl")]
    topic = np.array([r["topic"] for r in scores])
    rows = np.load(Path(config.gap_list_v2_dir) / "rows.npz")
    n_t, n_mix, n_ref, _ = rows["sizes"].tolist()
    assert len(prompts) == n_t == len(scores)

    # dolci chat rows and their scores, exactly as pooled_chat_clusters (plain variant)
    mix_text = pd.read_parquet(config.mix_text_path, columns=["text"])["text"].tolist()
    sample = np.sort(np.random.default_rng(config.seed).choice(len(mix_text), size=n_mix, replace=False))
    pos = {int(m): i for i, m in enumerate(sample)}
    chat = np.sort([pos[r["mix_index"]] for r in load_jsonl(config.chat_path)])
    chat_texts = [mix_text[sample[i]] for i in chat]
    del mix_text
    feats = np.load(config.feats_path)
    x = torch.tensor(feats, device="cuda").float()
    del feats
    xt, xmix, xdolma = x[:n_t], x[n_t : n_t + n_mix], x[n_t + n_mix : n_t + n_mix + n_ref]
    sd = xmix.std(0) + 1e-6
    xc = torch.tensor(chat, device="cuda")
    c_plain, _ = kth_distances(xmix / sd, (xmix / sd)[xc], config.k + 1, config.dup_extra, config.dup_eps, config.chunk)
    dolma_c = knn_distances(xdolma, xmix[xc], (config.k,))[config.k].cpu().numpy()
    xt_s = (xt / sd).cpu().numpy()
    del x, xt, xmix, xdolma
    torch.cuda.empty_cache()
    t_score = np.array([r["delta_knn"] for r in scores])
    score = np.r_[t_score, c_plain - dolma_c]
    is_traffic = np.r_[np.ones(n_t, dtype=bool), np.zeros(len(chat), dtype=bool)]
    n_pool = len(score)
    texts = prompts + chat_texts
    keys = np.array([template_key(t, config.prefix_chars) for t in texts], dtype=object)
    LOGGER.info(f"{n_pool=} {np.median(score[is_traffic])=:.4f} {np.median(score[~is_traffic])=:.4f}")

    # pooled groupings
    olmo_pool = np.load(Path(config.pooled_dir) / "pooled_cluster_labels.npy")
    assert len(olmo_pool) == n_pool
    k_pool = round(config.topic_count * n_pool / n_t)
    LOGGER.info("fitting tf-idf on the pool")
    q = TfidfVectorizer(sublinear_tf=True, min_df=2, ngram_range=(1, 2), max_features=200_000).fit_transform(texts)
    q = q.tocoo()
    qt = torch.sparse_coo_tensor(np.vstack([q.row, q.col]), q.data.astype(np.float32), q.shape, device="cuda")
    torch.manual_seed(config.seed)
    u, sv, _ = torch.svd_lowrank(qt, q=config.svd_dim, niter=4)
    z = (u * sv).cpu().numpy()
    del qt, u, sv
    z /= np.linalg.norm(z, axis=1, keepdims=True) + 1e-9
    LOGGER.info("tf-idf svd done")
    lab, _ = spherical_kmeans(torch.tensor(z, device="cuda"), k_pool, n_init=config.n_init, seed=config.seed)
    tfidf_pool = merge_small(lab.cpu().numpy(), z, config.min_cluster)
    np.save(out / "tfidf_pool_labels.npy", tfidf_pool)
    LOGGER.info(f"tf-idf clusters: {len(np.unique(tfidf_pool))=}")

    encoder = SentenceTransformer(config.encoder_id, device="cuda")
    emb = encoder.encode(texts, batch_size=256, normalize_embeddings=True, show_progress_bar=False)
    topic_ids = np.unique(topic)
    tcent = np.stack([emb[:n_t][topic == t].mean(0) for t in topic_ids])
    tcent /= np.linalg.norm(tcent, axis=1, keepdims=True)
    topic_pool = np.r_[topic, topic_ids[(emb[n_t:] @ tcent.T).argmax(1)]]
    random_pool = rng.permutation(olmo_pool)
    LOGGER.info("topics assigned")

    # traffic-only groupings, as in cluster_gap_list and tfidf_cluster_gap_list
    olmo_t = merge_small(np.load(Path(config.olmo_clusters_v2_dir) / "cluster_labels.npy"), xt_s, config.min_cluster)
    tfidf_t = np.load(Path(config.tfidf_dir) / "tfidf_cluster_labels.npy")
    assert len(olmo_t) == len(tfidf_t) == n_t
    LOGGER.info(f"{len(np.unique(olmo_t))=} {len(np.unique(tfidf_t))=} {len(topic_ids)=}")
    random_t = rng.permutation(olmo_t)

    groupings = {
        "olmo_clusters": (olmo_pool, olmo_t),
        "tfidf_clusters": (tfidf_pool, tfidf_t),
        "topics": (topic_pool, topic),
        "random_groups": (random_pool, random_t),
    }
    summary = {}
    for name, (pool_lab, t_lab) in groupings.items():
        gs = group_scores(score, pool_lab, keys)
        share = pd.Series(is_traffic.astype(float)).groupby(pool_lab).mean()
        gs["traffic_share"] = share.loc[gs.index].to_numpy()
        dolci_groups = gs.loc[gs["traffic_share"] < 0.5, "score"].to_numpy()
        traffic_groups = gs.loc[gs["traffic_share"] > 0.5, "score"].to_numpy()
        t_gs = group_scores(t_score, t_lab, keys[:n_t])["score"].to_numpy()
        boot_lines = np.stack(
            [np.quantile(rng.choice(dolci_groups, len(dolci_groups)), QS) for _ in range(config.n_boot)]
        )
        res = {
            "n_pool_groups": len(gs),
            "n_majority_dolci_groups": len(dolci_groups),
            "n_majority_traffic_groups": len(traffic_groups),
            "n_traffic_only_groups": len(t_gs),
        }
        for i, qv in enumerate(QS):
            line = float(np.quantile(dolci_groups, qv))
            pooled_ci = [int((traffic_groups > b).sum()) for b in boot_lines[:, i]]
            t_ci = [int((t_gs > b).sum()) for b in boot_lines[:, i]]
            res[str(qv)] = {
                "line": round(line, 4),
                "pooled_flagged": int((traffic_groups > line).sum()),
                "pooled_expected": round((1 - qv) * len(traffic_groups), 1),
                "pooled_flagged_ci95": [int(v) for v in np.percentile(pooled_ci, [2.5, 97.5])],
                "traffic_only_flagged": int((t_gs > line).sum()),
                "traffic_only_expected": round((1 - qv) * len(t_gs), 1),
                "traffic_only_flagged_ci95": [int(v) for v in np.percentile(t_ci, [2.5, 97.5])],
            }
        summary[name] = res
        LOGGER.info(f"{name}: {json.dumps(res)}")

    decision = {
        "n_traffic": n_t,
        "n_dolci_chat": len(chat),
        "k_pool": k_pool,
        "score_medians": {
            "traffic": round(float(np.median(score[is_traffic])), 4),
            "dolci_chat": round(float(np.median(score[~is_traffic])), 4),
        },
        "summary": summary,
    }
    write_json(decision, out / "decision.json")
    print(f"Results saved to {out / 'decision.json'}")
    print(json.dumps(decision, indent=1))


if __name__ == "__main__":
    main()
