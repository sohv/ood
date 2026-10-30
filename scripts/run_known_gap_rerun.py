# known_gap_rerun: adds the found gap (traffic prompts tagged pasted_document) to the dolci reference as known, rescores traffic, and reranks the olmo clusters to see which gaps surface next.
# uv run --no-sync -m scripts.run_known_gap_rerun --gap_list_v2_dir results/raw/261030_gap_list_v2 --olmo_clusters_v2_dir results/raw/261030_olmo_clusters_v2 --cluster_gap_list_dir results/raw/261030_cluster_gap_list_v2 --tags_dir data/processed/kind_groups --output_dir results/raw/261030_known_gap_rerun_v1 --model_id allenai/Olmo-3-7B-Instruct-DPO --seed 0

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
from src.generation.prompt_kinds import ALL_KINDS
from src.interp.train_only_detectors import knn_distances
from src.metrics.io import write_json, write_jsonl
from src.utils.config import Config, write_config_json
from src.utils.logging import setup_logging
from src.utils.seed import set_seed

LOGGER = logging.getLogger(__name__)
TOPS = (20, 50, 100)


@dataclass
class KnownGapConfig(Config):
    model_id: str = "allenai/Olmo-3-7B-Instruct-DPO"
    gap_list_v2_dir: str = ""
    gap_list_dir: str = "results/raw/261029_gap_list_v1"
    olmo_clusters_v2_dir: str = ""
    cluster_gap_list_dir: str = ""
    tags_dir: str = ""
    feats_path: str = "/dev/shm/gap_list_v2_feats.npy"
    known_kind: str = "pasted_document"
    k: int = 100
    min_cluster: int = 10
    prefix_chars: int = 40
    n_spot: int = 20
    n_examples: int = 5
    show_chars: int = 1200
    chunk: int = 2048
    seed: int = 0


def group_scores(score: np.ndarray, groups: np.ndarray, keys: list[str]) -> pd.DataFrame:
    frame = pd.DataFrame({"g": groups, "key": keys, "s": score})
    units = frame.groupby(["g", "key"])["s"].median()
    return pd.DataFrame({"score": units.groupby(level=0).median(), "n_prompts": frame.groupby("g").size()}).sort_values(
        "score", ascending=False, kind="stable"
    )


def main() -> None:
    config = simple_parsing.parse(KnownGapConfig)
    out = Path(config.output_dir)
    out.mkdir(parents=True, exist_ok=False)
    setup_logging(out)
    set_seed(config.seed)
    write_config_json(config, out)
    rng = np.random.default_rng(config.seed)

    scores = load_jsonl(Path(config.gap_list_v2_dir) / "prompt_scores.jsonl")
    prompts = [r["prompt"] for r in load_jsonl(Path(config.gap_list_dir) / "prompt_scores.jsonl")]
    tags = load_jsonl(Path(config.tags_dir) / "traffic_tags.jsonl")
    assert [t["id"] for t in tags] == [r["id"] for r in scores]
    rows = np.load(Path(config.gap_list_v2_dir) / "rows.npz")
    n_t, n_mix, n_ref, _ = rows["sizes"].tolist()
    keys = [template_key(p, config.prefix_chars) for p in prompts]
    known = np.array([t[config.known_kind] is True for t in tags])

    feats = np.load(config.feats_path)
    x = torch.tensor(feats, device="cuda").float()
    del feats
    xt, xmix, xdolma = x[:n_t], x[n_t : n_t + n_mix], x[n_t + n_mix : n_t + n_mix + n_ref]
    sd = xmix.std(0) + 1e-6
    xt_s, xmix_s = xt / sd, xmix / sd
    dolma_t = knn_distances(xdolma, xt, (config.k,))[config.k].cpu().numpy()
    before = knn_distances(xmix_s, xt_s, (config.k,), scale=False)[config.k].cpu().numpy() - dolma_t
    saved = np.array([r["delta_knn"] for r in scores])
    LOGGER.info(f"before vs saved: {np.abs(before - saved).max()=}")
    before = saved

    # reference = dolci sample plus the known-gap traffic prompts; known prompts skip themselves via the (k+1)-th neighbour
    ref = torch.cat([xmix_s, xt_s[torch.tensor(np.where(known)[0], device="cuda")]])
    d = knn_distances(ref, xt_s, (config.k, config.k + 1), scale=False)
    after = np.where(known, d[config.k + 1].cpu().numpy(), d[config.k].cpu().numpy()) - dolma_t

    # olmo clusters merged as in cluster_gap_list v2 (float32 scale)
    cl = np.load(Path(config.olmo_clusters_v2_dir) / "cluster_labels.npy")
    xt_np = xt_s.cpu().numpy()
    sizes = np.bincount(cl)
    big = np.where(sizes >= config.min_cluster)[0]
    cent = np.stack([xt_np[cl == c].mean(0) for c in big])
    cent /= np.linalg.norm(cent, axis=1, keepdims=True)
    small = np.where(sizes[cl] < config.min_cluster)[0]
    clusters = cl.copy()
    clusters[small] = big[(xt_np[small] / np.linalg.norm(xt_np[small], axis=1, keepdims=True) @ cent.T).argmax(1)]
    old = [
        r for r in load_jsonl(Path(config.cluster_gap_list_dir) / "groups.jsonl") if r["grouping"] == "olmo_clusters"
    ]
    old_rank = {r["group"]: r["rank"] for r in old}

    gs_b, gs_a = group_scores(before, clusters, keys), group_scores(after, clusters, keys)
    assert gs_b.index[:20].tolist() == [r["group"] for r in old[:20]]
    tag_frame = pd.DataFrame([{k: t[k] is True for k in ALL_KINDS} for t in tags])

    def top_stats(gs: pd.DataFrame) -> dict:
        out_ = {}
        for n in TOPS:
            sel = np.isin(clusters, gs.index[:n])
            out_[str(n)] = {
                "known_kind_share": round(float(known[sel].mean()), 4),
                "kind_shares": {k: round(float(tag_frame[k].to_numpy()[sel].mean()), 4) for k in ALL_KINDS},
                "new_vs_before_top": int(len(set(gs.index[:n]) - set(gs_b.index[:n]))),
            }
        return out_

    member = pd.Series(np.arange(n_t)).groupby(clusters)
    group_rows = []
    for rank, (g, r) in enumerate(gs_a.iloc[:200].iterrows()):
        idx = member.get_group(g).to_numpy()
        group_rows.append(
            {
                "id": f"olmo_clusters|{g}",
                "group": int(g),
                "rank_after": rank + 1,
                "rank_before": old_rank[int(g)],
                "score_after": round(float(r["score"]), 4),
                "score_before": round(float(gs_b.loc[g, "score"]), 4),
                "n_prompts": int(r["n_prompts"]),
                "known_kind_share": round(float(known[idx].mean()), 4),
                "examples": [prompts[j][:300] for j in idx[np.argsort(-after[idx])][:3]],
            }
        )
    write_jsonl(group_rows, out / "groups.jsonl")

    # spot check: new top-20 clusters that were not in the old top 20 (those were labelled in cluster_gap_list)
    new_top = [g for g in gs_a.index[: config.n_spot] if g not in set(gs_b.index[: config.n_spot])]
    spot, key = [], {}
    for i, g in enumerate(rng.permutation(new_top)):
        idx = member.get_group(g).to_numpy()
        pick = rng.choice(idx, size=min(config.n_examples, len(idx)), replace=False)
        sid = f"s{i:02d}"
        key[sid] = {"group": int(g), "rank_after": int(gs_a.index.get_loc(g)) + 1}
        spot.append({"id": sid, "order": i, "examples": [prompts[j][: config.show_chars] for j in pick]})
    write_jsonl(spot, out / "spot_items.jsonl")
    write_json(key, out / "spot_key.json")

    decision = {
        "n_traffic": n_t,
        "n_known": int(known.sum()),
        "known_kind": config.known_kind,
        "score_change_known_median": round(float(np.median(after[known] - before[known])), 4),
        "score_change_other_median": round(float(np.median(after[~known] - before[~known])), 4),
        "before": top_stats(gs_b),
        "after": top_stats(gs_a),
        "n_new_in_top20": len(new_top),
        "rank_spearman_before_after": round(
            float(gs_b["score"].rank().corr(gs_a["score"].reindex(gs_b.index).rank(), method="spearman")), 4
        ),
    }
    write_json(decision, out / "decision.json")
    print(f"Results saved to {out / 'decision.json'}, {out / 'groups.jsonl'} and {out / 'spot_items.jsonl'}")
    print(json.dumps({k: v for k, v in decision.items() if k not in ("before", "after")}, indent=1))
    print(
        json.dumps(
            {
                "before_top50": decision["before"]["50"]["known_kind_share"],
                "after_top50": decision["after"]["50"]["known_kind_share"],
            }
        )
    )


if __name__ == "__main__":
    main()
