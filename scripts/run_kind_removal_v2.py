# kind_removal_v2: removes each prompt kind from the 200k dolci sample, rescores traffic and held-out dolci, and compares specificity and recovery by grouping against random and cluster removals.
# uv run --no-sync -m scripts.run_kind_removal_v2 --gap_list_v2_dir results/raw/261030_gap_list_v2 --olmo_clusters_v2_dir results/raw/261030_olmo_clusters_v2 --tags_dir data/processed/kind_groups --output_dir results/raw/261030_kind_removal_v2 --model_id allenai/Olmo-3-7B-Instruct-DPO --seed 0

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import simple_parsing
import torch

from scripts.run_gap_list_v2 import length_pct, template_key
from src.data.io import load_jsonl
from src.generation.prompt_kinds import ALL_KINDS
from src.interp.kmeans import spherical_kmeans
from src.interp.train_only_detectors import knn_distances
from src.metrics.io import write_json, write_jsonl
from src.utils.config import Config, write_config_json
from src.utils.logging import setup_logging
from src.utils.seed import set_seed

LOGGER = logging.getLogger(__name__)


@dataclass
class KindRemovalV2Config(Config):
    model_id: str = "allenai/Olmo-3-7B-Instruct-DPO"
    gap_list_v2_dir: str = ""
    olmo_clusters_v2_dir: str = ""
    gap_list_dir: str = "results/raw/261029_gap_list_v1"
    tags_dir: str = ""
    feats_path: str = "/dev/shm/gap_list_v2_feats.npy"
    k: int = 100
    min_removed: int = 500
    n_random: int = 10
    n_dolci_clusters: int = 200
    top_share: float = 0.05
    prefix_chars: int = 40
    seed: int = 0


def main() -> None:
    config = simple_parsing.parse(KindRemovalV2Config)
    out = Path(config.output_dir)
    out.mkdir(parents=True, exist_ok=False)
    setup_logging(out)
    set_seed(config.seed)
    write_config_json(config, out)
    torch.backends.cudnn.deterministic = True

    scores = load_jsonl(Path(config.gap_list_v2_dir) / "prompt_scores.jsonl")
    prompts = [r["prompt"] for r in load_jsonl(Path(config.gap_list_dir) / "prompt_scores.jsonl")]
    t_tags = load_jsonl(Path(config.tags_dir) / "traffic_tags.jsonl")
    d_tags = load_jsonl(Path(config.tags_dir) / "dolci_tags.jsonl")
    assert [t["id"] for t in t_tags] == [r["id"] for r in scores]
    rows = np.load(Path(config.gap_list_v2_dir) / "rows.npz")
    n_t, n_mix, n_ref, n_h = rows["sizes"].tolist()
    assert len(d_tags) == n_mix
    len_t, len_h = rows["lengths"][:n_t], rows["lengths"][n_t + n_mix + n_ref :]
    t_is = {k: np.array([t[k] is True for t in t_tags]) for k in ALL_KINDS}
    # a null dolci tag counts as the kind, as in chat_removal_loop
    d_is = {k: np.array([t[k] is not False for t in d_tags]) for k in ALL_KINDS}
    frame = pd.DataFrame(
        {
            "topic": [r["topic"] for r in scores],
            "cluster": np.load(Path(config.olmo_clusters_v2_dir) / "cluster_labels.npy"),
            "key": [template_key(p, config.prefix_chars) for p in prompts],
        }
    )

    feats = np.load(config.feats_path)
    x = torch.tensor(feats, device="cuda").float()
    del feats
    xt, xmix = x[:n_t], x[n_t : n_t + n_mix]
    xdolma, xh = x[n_t + n_mix : n_t + n_mix + n_ref], x[n_t + n_mix + n_ref :]
    # one fixed scaling for every arm: the full sample's per-dimension std
    sd = xmix.std(0) + 1e-6
    xt_s, xh_s, xmix_s = xt / sd, xh / sd, xmix / sd
    dolma_t = knn_distances(xdolma, xt, (config.k,))[config.k].cpu().numpy()
    dolma_h = knn_distances(xdolma, xh, (config.k,))[config.k].cpu().numpy()
    del x, xt, xh, xmix, xdolma
    labels, _ = spherical_kmeans(xmix_s, config.n_dolci_clusters, n_init=1, seed=config.seed)
    dolci_cluster = labels.cpu().numpy()
    torch.cuda.empty_cache()

    def rescore(keep: np.ndarray) -> np.ndarray:
        ref = xmix_s[torch.tensor(keep, device="cuda")]
        s_t = knn_distances(ref, xt_s, (config.k,), scale=False)[config.k].cpu().numpy() - dolma_t
        s_h = knn_distances(ref, xh_s, (config.k,), scale=False)[config.k].cpu().numpy() - dolma_h
        return length_pct(s_t, len_t, s_h, len_h)

    def group_scores(pct: np.ndarray, by: str) -> pd.Series:
        units = frame.assign(pct=pct).groupby([by, "key"])["pct"].median()
        return units.groupby(level=0).median()

    sizes = {by: frame.groupby(by).size() for by in ("topic", "cluster")}
    n_top_prompts = int(config.top_share * n_t)

    def recovery(pct: np.ndarray, removed_kind: str) -> dict:
        # share of the removed kind's traffic prompts inside the top groups holding 5% of traffic
        out_ = {}
        target = t_is[removed_kind]
        for by in ("topic", "cluster"):
            order = group_scores(pct, by).sort_values(ascending=False).index
            held = sizes[by].loc[order].cumsum()
            top = set(order[: int((held < n_top_prompts).sum()) + 1])
            out_[by] = float(target[frame[by].isin(top).to_numpy()].sum() / target.sum())
        kmed = sorted(((float(np.median(pct[t_is[k]])), k) for k in ALL_KINDS if t_is[k].any()), reverse=True)
        out_["kind_rank"] = [k for _, k in kmed].index(removed_kind) + 1
        return out_

    def kind_medians(pct: np.ndarray) -> dict:
        return {k: float(np.median(pct[t_is[k]])) for k in ALL_KINDS if t_is[k].any()}

    full = rescore(np.ones(n_mix, dtype=bool))
    gap_list_v2 = np.array([r["length_pct"] for r in scores])
    LOGGER.info(f"full arm vs gap_list_v2 max abs diff {np.abs(full - gap_list_v2).max():.5f}")
    full_med, rows_out, eligible = kind_medians(full), [], []
    for kind in ALL_KINDS:
        removed = d_is[kind]
        n_rm = int(removed.sum())
        if n_rm < config.min_removed or not t_is[kind].any():
            LOGGER.info(f"{kind}: skipped, {n_rm} dolci rows tagged")
            continue
        eligible.append(kind)
        arms = {"full": np.ones(n_mix, dtype=bool), "removed": ~removed}
        for s in range(config.n_random):
            keep = np.ones(n_mix, dtype=bool)
            keep[np.random.default_rng(s).choice(n_mix, size=n_rm, replace=False)] = False
            arms[f"random{s}"] = keep
        # whole dolci clusters, added in random order until the removal size is reached
        keep, taken = np.ones(n_mix, dtype=bool), 0
        for c in np.random.default_rng(config.seed).permutation(config.n_dolci_clusters):
            if taken >= n_rm:
                break
            members = np.where(dolci_cluster == c)[0][: n_rm - taken]
            keep[members] = False
            taken += len(members)
        arms["cluster"] = keep
        for arm, keep in arms.items():
            pct = full if arm == "full" else rescore(keep)
            med = kind_medians(pct)
            rows_out.append(
                {
                    "id": f"{kind}|{arm}",
                    "removed_kind": kind,
                    "arm": arm,
                    "n_removed": int((~keep).sum()),
                    "shift": {k: round(med[k] - full_med[k], 5) for k in med},
                    "recovery": recovery(pct, kind),
                }
            )
        LOGGER.info(f"{kind}: removed {n_rm}, {rows_out[-12]['recovery']=} {rows_out[-11]['recovery']=}")
        torch.cuda.empty_cache()
    write_jsonl(rows_out, out / "arms.jsonl")

    table, grouping = {}, {}
    for kind in eligible:
        r = {x["arm"]: x for x in rows_out if x["removed_kind"] == kind}
        rand = [r[f"random{s}"] for s in range(config.n_random)]
        z = {}
        for m in full_med:
            mu, sdv = np.mean([a["shift"][m] for a in rand]), np.std([a["shift"][m] for a in rand]) + 1e-9
            z[m] = round(float((r["removed"]["shift"][m] - mu) / sdv), 2)
        table[kind] = z
        grouping[kind] = {arm: r[arm]["recovery"] for arm in ("full", "removed", "cluster")} | {
            "random_mean": {
                by: round(float(np.mean([a["recovery"][by] for a in rand])), 4) for by in ("topic", "cluster")
            },
            "random_max": {
                by: round(float(np.max([a["recovery"][by] for a in rand])), 4) for by in ("topic", "cluster")
            },
            "random_kind_ranks": [a["recovery"]["kind_rank"] for a in rand],
        }
    decision = {
        "n_traffic": n_t,
        "n_heldout": n_h,
        "full_vs_gap_list_v2_max_abs_diff": round(float(np.abs(full - gap_list_v2).max()), 6),
        "n_removed": {k: int(d_is[k].sum()) for k in ALL_KINDS},
        "n_null_dolci_tags": {k: int(sum(t[k] is None for t in d_tags)) for k in ALL_KINDS},
        "eligible": eligible,
        "n_rows": len(rows_out),
        "n_rows_expected": len(eligible) * (config.n_random + 3),
        "specificity_z": table,
        "grouping": grouping,
    }
    write_json(decision, out / "decision.json")
    print(f"Results saved to {out / 'arms.jsonl'} and {out / 'decision.json'}")
    for k in eligible:
        print(
            k,
            "own z",
            table[k][k],
            "| removed",
            grouping[k]["removed"],
            "| random",
            grouping[k]["random_mean"],
            "| cluster",
            grouping[k]["cluster"],
        )


if __name__ == "__main__":
    main()
