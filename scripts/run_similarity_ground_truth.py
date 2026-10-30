# cluster_ground_truth: labels every gap-search topic by its prompts' similarity to their nearest mix prompts, checks the label once against the user's labels, then reports precision at k.
# uv run -m scripts.run_similarity_ground_truth --run_dir results/raw/261026_cluster_ground_truth_similarity_gt_v1 --pipeline_million_dir results/raw/261021_pipeline_million_gap_pipeline_v2 --must_not_flag_set_dir results/raw/261026_must_not_flag_set_eval_set_v1 --label_files results/raw/261022_deep_gap_loop_topic_labels_v1/decision.json results/raw/261022_classifier_gap_loop_topic_labels_v1/decision.json --seed 0

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import simple_parsing
import torch
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from src.data.gap_pool import load_mix, load_traffic_pool
from src.data.io import load_jsonl
from src.metrics.gap_pipeline import text_hash
from src.metrics.io import write_json, write_jsonl
from src.utils.git import get_git_hash
from src.utils.logging import setup_logging
from src.utils.seed import set_seed

LOGGER = logging.getLogger(__name__)
CLASSES = ("deployment_only", "shared", "covered")


@dataclass
class SimilarityGtConfig:
    run_dir: str = ""
    pipeline_million_dir: str = ""
    must_not_flag_set_dir: str = ""
    label_files: list[str] = field(default_factory=list)
    seed: int = 0
    headline_low: float = 20.0
    covered_pct: float = 50.0
    alt_lows: list[float] = field(default_factory=lambda: [10.0, 30.0])
    hi_cos: float = 0.80
    ks: list[int] = field(default_factory=lambda: [10, 20, 50, 100, 200, 293])
    chunk: int = 2048


def nearest_mix_cosine(
    t_np: np.ndarray, t_hash: np.ndarray, m_np: np.ndarray, m_hash: np.ndarray, chunk: int
) -> np.ndarray:
    # the most similar mix prompt per traffic prompt; an exact copy (same text hash) yields to the second nearest
    m_gpu, mh_gpu = torch.from_numpy(m_np).cuda(), torch.from_numpy(m_hash).cuda()
    out = np.empty(len(t_np), dtype=np.float32)
    for s in range(0, len(t_np), chunk):
        q = torch.from_numpy(t_np[s : s + chunk]).cuda()
        qh = torch.from_numpy(t_hash[s : s + chunk]).cuda()
        sims, idx = (q @ m_gpu.T).float().topk(2, dim=1)
        same = mh_gpu[idx[:, 0]] == qh
        out[s : s + chunk] = torch.where(same, sims[:, 1], sims[:, 0]).cpu().numpy()
    return out


def main():
    config = simple_parsing.parse(SimilarityGtConfig)
    assert (
        config.run_dir and config.pipeline_million_dir and config.must_not_flag_set_dir and config.label_files
    ), "paths required"
    run = Path(config.run_dir)
    assert not run.exists(), f"{run} exists; results/raw is append-only"
    setup_logging(run)
    set_seed(config.seed)
    write_json({"git_hash": get_git_hash(), **asdict(config)}, run / "config.json")

    c37 = json.loads((Path(config.pipeline_million_dir) / "config.json").read_text())
    rows, t_np, _, _ = load_traffic_pool(
        c37["iterative_discovery_dir"],
        c37["self_nature_a_dir"],
        c37["emb_v2"],
        c37["new_lmsys_shards"],
        c37["embed_chars"],
    )
    t_hash = np.array([text_hash(r["norm"]) for r in rows], dtype=np.int64)
    labels = np.load(Path(config.pipeline_million_dir) / "labels.npy")
    m_np, m_hash = load_mix(c37["sft_path"], c37["emb_v1"])
    assert len(t_np) == 992933 and len(m_np) == 1710578, (len(t_np), len(m_np))
    clusters = {int(c["id"][1:]): c for c in load_jsonl(Path(config.pipeline_million_dir) / "clusters.jsonl")}
    ranked = sorted((c for c in clusters if clusters[c]["rank"]), key=lambda c: clusters[c]["rank"])

    near = nearest_mix_cosine(t_np, t_hash, m_np, m_hash, config.chunk)
    np.save(run / "nearest_mix_cosine.npy", near)
    LOGGER.info(
        f"nearest-mix cosine: median {np.median(near):.3f}, share at or above {config.hi_cos}: {np.mean(near >= config.hi_cos):.3f}"
    )

    sim_med = np.array([float(np.median(near[labels == c])) for c in ranked])
    share_hi = np.array([float(np.mean(near[labels == c] >= config.hi_cos)) for c in ranked])
    cuts = {f"P{config.headline_low:g}": float(np.percentile(sim_med, config.headline_low))}
    for p in config.alt_lows:
        cuts[f"P{p:g}"] = float(np.percentile(sim_med, p))
    covered_cut = float(np.percentile(sim_med, config.covered_pct))

    def classify(s: float, low: float) -> str:
        return "deployment_only" if s <= low else "covered" if s >= covered_cut else "shared"

    headline = f"P{config.headline_low:g}"
    ratio = np.array([clusters[c]["ratio"] for c in ranked])
    e55 = {r["id"]: r for r in load_jsonl(Path(config.must_not_flag_set_dir) / "cluster_ground_truth.jsonl")}
    mix_share = np.array([e55[f"c{c}"]["mix_share"] for c in ranked])

    human = {}
    for path in config.label_files:
        for r in json.loads(Path(path).read_text())["rows"]:
            human[int(r["id"][1:])] = r["human"]
    pos = {c: i for i, c in enumerate(ranked)}
    assert all(c in pos for c in human), "a labelled topic is not a ranked cluster"
    agree, scored = 0, 0
    confusion = {h: {cl: 0 for cl in CLASSES} for h in ("real_gap", "partly", "covered")}
    for c, h in human.items():
        cl = classify(sim_med[pos[c]], cuts[headline])
        confusion[h][cl] += 1
        if h == "partly":
            continue
        scored += 1
        agree += (h == "real_gap" and cl == "deployment_only") or (h == "covered" and cl != "deployment_only")
    agreement = agree / max(1, scored)
    valid = agreement >= 0.70
    y = [1 if human[c] == "real_gap" else 0 for c in human if human[c] != "partly"]
    s = [1 - sim_med[pos[c]] for c in human if human[c] != "partly"]
    auc = float(roc_auc_score(y, s))

    precision = {
        name: {
            str(K): float(np.mean([classify(sim_med[i], low) == "deployment_only" for i in range(K)]))
            for K in config.ks
        }
        for name, low in cuts.items()
    }
    counts = {
        name: {cl: int(sum(classify(x, low) == cl for x in sim_med)) for cl in CLASSES} for name, low in cuts.items()
    }
    p100 = precision[headline]["100"]
    usable = "usable" if p100 >= 0.5 else "not_usable" if p100 < 0.3 else "ambiguous"

    write_jsonl(
        [
            {
                "id": f"c{c}",
                "rank": clusters[c]["rank"],
                "ratio": clusters[c]["ratio"],
                "sim_med": round(float(sim_med[i]), 4),
                "share_hi": round(float(share_hi[i]), 4),
                **{f"class_{name}": classify(sim_med[i], low) for name, low in cuts.items()},
                "human": human.get(c),
            }
            for i, c in enumerate(ranked)
        ],
        run / "topic_similarity.jsonl",
    )
    decision = {
        "git_hash": get_git_hash(),
        "cuts": {**cuts, "covered": covered_cut},
        "label_valid": bool(valid),
        "human_agreement": {
            "agreement": agreement,
            "n_scored": scored,
            "n_labelled": len(human),
            "confusion_at_headline": confusion,
        },
        "label_auc_real_vs_covered": auc,
        "precision_at_k": precision,
        "usable_by_stefans_line": usable if valid else "unvalidated",
        "class_counts": counts,
        "spearman_sim_vs_centroid_share": float(spearmanr(sim_med, mix_share).statistic),
        "spearman_sim_vs_neighbour_ratio": float(spearmanr(sim_med, ratio).statistic),
        "sanity": {
            "n_traffic": len(t_np),
            "n_mix": len(m_np),
            "ranked_topics": len(ranked),
            "labelled_ranked": len(human),
        },
    }
    write_json(decision, run / "decision.json")
    print(f"label valid {valid}: agreement {agreement:.3f} on {scored}; AUC real vs covered {auc:.3f}")
    print(f"confusion {confusion}")
    print(
        f"precision@K at {headline}: {precision[headline]}; {decision['usable_by_stefans_line']}; counts {counts[headline]}"
    )
    print(f"alt cuts { {k: v for k, v in precision.items() if k != headline} }")
    print(
        f"spearman vs centroid share {decision['spearman_sim_vs_centroid_share']:.3f}, vs neighbour ratio {decision['spearman_sim_vs_neighbour_ratio']:.3f}"
    )
    print(f"decision written to {run / 'decision.json'}")


if __name__ == "__main__":
    main()
