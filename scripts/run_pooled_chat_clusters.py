# pooled_chat_clusters: clusters traffic together with dolci's own real-chat rows in olmo space and asks whether the clusters over the gap line are mostly traffic or just as much messy-but-covered dolci chat.
# uv run --no-sync -m scripts.run_pooled_chat_clusters --gap_list_v2_dir results/raw/261030_gap_list_v2 --chat_path data/processed/chat_kinds/chat_kinds.jsonl --output_dir results/raw/261030_pooled_chat_clusters_v1 --model_id allenai/Olmo-3-7B-Instruct-DPO --seed 0

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import simple_parsing
import torch

from scripts.run_gap_list_v2 import template_key
from src.data.io import load_jsonl
from src.interp.kmeans import spherical_kmeans
from src.interp.train_only_detectors import knn_distances
from src.metrics.io import write_json, write_jsonl
from src.utils.config import Config, write_config_json
from src.utils.logging import setup_logging
from src.utils.seed import set_seed

LOGGER = logging.getLogger(__name__)
UNIT_BINS = (0, 10, 20, 40, 80, np.inf)
QS = (0.95, 0.99)
TOPS = (20, 50, 100, 200)


@dataclass
class PooledConfig(Config):
    model_id: str = "allenai/Olmo-3-7B-Instruct-DPO"
    gap_list_v2_dir: str = ""
    gap_list_dir: str = "results/raw/261029_gap_list_v1"
    chat_path: str = ""
    mix_text_path: str = "cache/safety_removal/mix_text.parquet"
    feats_path: str = "/dev/shm/gap_list_v2_feats.npy"
    k: int = 100
    dup_eps: float = 0.05
    dup_extra: int = 50
    topic_count: int = 1903
    min_cluster: int = 10
    n_init: int = 3
    prefix_chars: int = 40
    n_null: int = 2000
    n_boot: int = 2000
    chunk: int = 2048
    seed: int = 0


def kth_distances(ref: torch.Tensor, x: torch.Tensor, k: int, extra: int, eps: float, chunk: int) -> tuple:
    # k-th nearest cosine distance, plus the same after dropping neighbours closer than eps (self and near copies)
    ref, x = torch.nn.functional.normalize(ref, dim=1), torch.nn.functional.normalize(x, dim=1)
    plain, dedup = [], []
    for s in range(0, len(x), chunk):
        d = torch.sqrt(torch.clamp(2 - 2 * (x[s : s + chunk] @ ref.T).topk(k + extra, dim=1).values, min=0))
        plain.append(d[:, k - 1])
        far = d >= eps
        pos = torch.clamp((far.cumsum(1) >= k).float().argmax(1), max=k + extra - 1)
        dedup.append(d.gather(1, pos[:, None])[:, 0])
    return torch.cat(plain).cpu().numpy(), torch.cat(dedup).cpu().numpy()


def group_scores(score: np.ndarray, groups: np.ndarray, keys: np.ndarray) -> pd.DataFrame:
    frame = pd.DataFrame({"g": groups, "key": keys, "s": score})
    units = frame.groupby(["g", "key"])["s"].median()
    return pd.DataFrame(
        {
            "score": units.groupby(level=0).median(),
            "n_units": units.groupby(level=0).size(),
            "n_prompts": frame.groupby("g").size(),
        }
    ).sort_values("score", ascending=False, kind="stable")


def main() -> None:
    config = simple_parsing.parse(PooledConfig)
    out = Path(config.output_dir)
    out.mkdir(parents=True, exist_ok=False)
    setup_logging(out)
    set_seed(config.seed)
    write_config_json(config, out)
    rng = np.random.default_rng(config.seed)
    torch.backends.cudnn.deterministic = True

    scores = load_jsonl(Path(config.gap_list_v2_dir) / "prompt_scores.jsonl")
    prompts = [r["prompt"] for r in load_jsonl(Path(config.gap_list_dir) / "prompt_scores.jsonl")]
    rows = np.load(Path(config.gap_list_v2_dir) / "rows.npz")
    n_t, n_mix, n_ref, _ = rows["sizes"].tolist()
    assert len(prompts) == n_t == len(scores)

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
    xt_s, xmix_s = xt / sd, xmix / sd
    xc = torch.tensor(chat, device="cuda")

    # dolci chat rows are part of the reference, so their plain score uses the (k+1)-th neighbour to skip themselves
    t_plain, t_dedup = kth_distances(xmix_s, xt_s, config.k, config.dup_extra, config.dup_eps, config.chunk)
    c_plain, _ = kth_distances(xmix_s, xmix_s[xc], config.k + 1, config.dup_extra, config.dup_eps, config.chunk)
    _, c_dedup = kth_distances(xmix_s, xmix_s[xc], config.k, config.dup_extra, config.dup_eps, config.chunk)
    dolma_t = knn_distances(xdolma, xt, (config.k,))[config.k].cpu().numpy()
    dolma_c = knn_distances(xdolma, xmix[xc], (config.k,))[config.k].cpu().numpy()
    saved = np.array([r["delta_knn"] for r in scores])
    LOGGER.info(f"traffic plain vs saved: {np.abs(t_plain - dolma_t - saved).max()=}")
    variants = {
        "plain": np.r_[t_plain - dolma_t, c_plain - dolma_c],
        "dedup": np.r_[t_dedup - dolma_t, c_dedup - dolma_c],
    }

    # one clustering of the pooled prompts, cluster count scaled from the topic count, small clusters merged
    is_traffic = np.r_[np.ones(n_t, dtype=bool), np.zeros(len(chat), dtype=bool)]
    n_pool = len(is_traffic)
    xp = torch.cat([xt_s, xmix_s[xc]])
    k_pool = round(config.topic_count * n_pool / n_t)
    lab, _ = spherical_kmeans(xp, k_pool, n_init=config.n_init, seed=config.seed)
    lab = lab.cpu().numpy()
    del x, xt, xmix, xdolma
    xp_n = torch.nn.functional.normalize(xp, dim=1)
    sizes = np.bincount(lab, minlength=k_pool)
    big = np.where(sizes >= config.min_cluster)[0]
    cent = torch.nn.functional.normalize(
        torch.stack([xp_n[torch.tensor(np.where(lab == c)[0], device="cuda")].mean(0) for c in big]), dim=1
    )
    small = np.where(sizes[lab] < config.min_cluster)[0]
    groups = lab.copy()
    groups[small] = big[(xp_n[torch.tensor(small, device="cuda")] @ cent.T).argmax(1).cpu().numpy()]
    np.save(out / "pooled_cluster_labels.npy", groups)
    texts = prompts + chat_texts
    keys = np.array([template_key(t, config.prefix_chars) for t in texts], dtype=object)
    base = float(is_traffic.mean())
    member = pd.Series(np.arange(n_pool)).groupby(groups)

    summary, group_rows = {}, []
    for name, score in variants.items():
        gs = group_scores(score, groups, keys)

        def random_group_score(pick: np.ndarray) -> float:
            return float(pd.Series(score[pick]).groupby(keys[pick]).median().median())

        bins = np.digitize(gs["n_units"], UNIT_BINS[1:-1], right=True)
        cuts = {}
        for b in np.unique(bins):
            size = int(np.median(gs["n_prompts"][bins == b]))
            null = [random_group_score(rng.choice(n_pool, size=size, replace=False)) for _ in range(config.n_null)]
            cuts[int(b)] = {q: float(np.quantile(null, q)) for q in QS}
        gs["traffic_share"] = [float(is_traffic[member.get_group(g)].mean()) for g in gs.index]
        for q in QS:
            gs[f"flag_{q}"] = [s > cuts[int(b)][q] for s, b in zip(gs["score"], bins, strict=True)]

        def flagged_share(sel: pd.DataFrame) -> dict:
            # prompt-weighted traffic share in the selected clusters, with a cluster bootstrap interval
            if not len(sel):
                return {"n_clusters": 0}
            t = (sel["traffic_share"] * sel["n_prompts"]).to_numpy()
            n = sel["n_prompts"].to_numpy()
            boot = [
                (t[i].sum() / n[i].sum()) for i in (rng.integers(0, len(sel), len(sel)) for _ in range(config.n_boot))
            ]
            return {
                "n_clusters": len(sel),
                "n_prompts": int(n.sum()),
                "traffic_share": round(float(t.sum() / n.sum()), 4),
                "ci95": [round(float(v), 4) for v in np.percentile(boot, [2.5, 97.5])],
                "majority_traffic_clusters": int((sel["traffic_share"] > 0.5).sum()),
                "majority_dolci_chat_clusters": int((sel["traffic_share"] < 0.5).sum()),
            }

        summary[name] = {
            "n_clusters": len(gs),
            "flagged": {str(q): flagged_share(gs[gs[f"flag_{q}"]]) for q in QS},
            "top": {str(n): flagged_share(gs.iloc[:n]) for n in TOPS},
            "all_clusters_majority_traffic": int((gs["traffic_share"] > 0.5).sum()),
        }
        for rank, (g, r) in enumerate(gs.iloc[:200].iterrows()):
            idx = member.get_group(g).to_numpy()
            group_rows.append(
                {
                    "id": f"{name}|{g}",
                    "variant": name,
                    "group": int(g),
                    "rank": rank + 1,
                    "score": round(float(r["score"]), 4),
                    "n_prompts": int(r["n_prompts"]),
                    "traffic_share": round(float(r["traffic_share"]), 4),
                    **{f"flag_{q}": bool(r[f"flag_{q}"]) for q in QS},
                    "examples": [
                        {"source": "traffic" if is_traffic[j] else "dolci_chat", "text": texts[j][:300]}
                        for j in idx[np.argsort(-score[idx])][:4]
                    ],
                }
            )
    write_jsonl(group_rows, out / "groups.jsonl")
    decision = {
        "n_traffic": n_t,
        "n_dolci_chat": len(chat),
        "pool_traffic_share": round(base, 4),
        "k_pool": k_pool,
        "n_small_merged": len(small),
        "score_medians": {
            v: {
                "traffic": round(float(np.median(s[is_traffic])), 4),
                "dolci_chat": round(float(np.median(s[~is_traffic])), 4),
            }
            for v, s in variants.items()
        },
        "summary": summary,
    }
    write_json(decision, out / "decision.json")
    print(f"Results saved to {out / 'decision.json'} and {out / 'groups.jsonl'}")
    print(json.dumps(decision, indent=1))


if __name__ == "__main__":
    main()
