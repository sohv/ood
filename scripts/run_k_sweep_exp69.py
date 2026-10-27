# exp69: scores exp52's and exp61's topics with the neighbour count at k = 50, 100 and 200 from one CPU pass and asks whether a larger k predicts over-refusal or the user's labels better than k = 50.
# uv run -m scripts.run_k_sweep_exp69 --run_dir results/raw/261027_exp69_k_sweep_v1 --exp37_dir results/raw/261021_exp37_gap_pipeline_v2 --exp62_dir results/raw/261026_exp62_behaviour_predictors_v1 --exp61_dir results/raw/261026_exp61_labels_v1 --seed 0

import os

os.environ.setdefault("OMP_NUM_THREADS", "96")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "96")

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import simple_parsing
import torch
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from src.data.gap_pool import load_mix, load_traffic_pool
from src.data.io import load_jsonl
from src.metrics.gap_pipeline import density_ratios_multi_k, text_hash
from src.metrics.io import write_json, write_jsonl
from src.utils.git import get_git_hash
from src.utils.logging import setup_logging
from src.utils.seed import set_seed

LOGGER = logging.getLogger(__name__)
KS = (50, 100, 200)


@dataclass
class KSweepConfig:
    run_dir: str = ""
    exp37_dir: str = ""
    exp62_dir: str = ""
    exp61_dir: str = ""
    seed: int = 0
    line_sample: int = 50000
    n_random_subsets: int = 1000
    subset_size: int = 500
    ratio_multiple: float = 3.0
    over_refused: float = 0.20
    n_boot: int = 2000
    margin: float = 0.10
    ci_level: float = 0.975
    repro_tol: float = 1.0 + 1e-6
    repro_share: float = 0.99
    repro_min_spearman: float = 0.98
    num_tasks: int | None = None


def main():
    config = simple_parsing.parse(KSweepConfig)
    assert config.run_dir and config.exp37_dir and config.exp62_dir and config.exp61_dir
    run = Path(config.run_dir)
    assert not run.exists(), f"{run} exists; results/raw is append-only"
    setup_logging(run)
    set_seed(config.seed)
    torch.set_num_threads(int(os.environ["OMP_NUM_THREADS"]))
    write_json({"git_hash": get_git_hash(), **asdict(config)}, run / "config.json")
    rng = np.random.default_rng(config.seed)

    c37 = json.loads((Path(config.exp37_dir) / "config.json").read_text())
    rows, t_np, _, _ = load_traffic_pool(
        c37["exp09_dir"], c37["exp31_dir"], c37["emb_v2"], c37["new_lmsys_shards"], c37["embed_chars"]
    )
    t_hash = np.array([text_hash(r["norm"]) for r in rows], dtype=np.int64)
    labels = np.load(Path(config.exp37_dir) / "labels.npy")
    ratio37 = np.load(Path(config.exp37_dir) / "ratio.npy")
    m_np, m_hash = load_mix(c37["sft_path"], c37["emb_v1"])[:2]
    t_np = t_np.astype(np.float32)
    LOGGER.info(f"traffic {t_np.shape}, mix {m_np.shape}")

    # topics: exp52's 221 with refusal effects, plus exp61's labelled topics
    e62 = {int(r["id"][1:]): r for r in load_jsonl(Path(config.exp62_dir) / "topics.jsonl")}
    e61 = json.loads((Path(config.exp61_dir) / "decision.json").read_text())
    label = {int(r["id"][1:]): r["label"] for r in e61["new_rows"]} | {
        int(r["id"][1:]): r["second"] for r in e61["relabel_rows"]
    }
    topics = sorted(set(e62) | set(label))
    if config.num_tasks:
        topics = topics[: config.num_tasks]
        config.line_sample = min(config.line_sample, 2000)
    members = {c: np.where(labels == c)[0] for c in topics}
    scored = np.concatenate([members[c] for c in topics])
    line_sample = rng.choice(len(rows), size=config.line_sample, replace=False)
    LOGGER.info(f"{len(topics)} topics, {len(scored)} member prompts, {len(line_sample)} line-sample prompts")

    t_t, th_t = torch.from_numpy(t_np), torch.from_numpy(t_hash)
    m_t, mh_t = torch.from_numpy(m_np.astype(np.float32)), torch.from_numpy(m_hash)

    def ratios(idx: np.ndarray) -> dict[int, np.ndarray]:
        return density_ratios_multi_k(t_np[idx], t_hash[idx], idx, t_t, th_t, m_t, mh_t, ks=KS)

    r_line = ratios(line_sample)
    LOGGER.info("line sample scored")
    r_scored = ratios(scored)
    LOGGER.info("members scored")

    # per-k gap line, exp64's rule, and the ceiling share of the random sample
    lines, ceiling = {}, {}
    for k in KS:
        sub_rng = np.random.default_rng(config.seed)
        medians = [
            float(np.median(r_line[k][sub_rng.choice(len(r_line[k]), size=config.subset_size, replace=False)]))
            for _ in range(config.n_random_subsets)
        ]
        lines[k] = config.ratio_multiple * float(np.median(medians))
        cap = ((k + 1) / len(t_np)) / (1 / len(m_np))
        ceiling[k] = float(np.mean(r_line[k] >= cap - 1e-6))
    LOGGER.info(f"lines {lines}; ceiling shares {ceiling}")

    # topic medians at each k, and the k = 50 reproduction of exp37
    pos = np.empty(len(t_np), dtype=np.int64)
    pos[scored] = np.arange(len(scored))
    med = {k: {c: float(np.median(r_scored[k][pos[members[c]]])) for c in topics} for k in KS}
    med37 = {c: float(np.median(ratio37[members[c]])) for c in topics}

    # amendment: exp37's gpu fp16 pass differs at the k-th neighbour on some rows, so agreement is within one neighbour
    def implied_traffic(ratio: float, k: int = 50) -> float:
        # invert ((t + 1) / n_t) / ((m + 1) / n_m) with t + m = k for the traffic count t
        r = ratio * len(t_np) / len(m_np)
        return ((k + 1) * r - 1) / (1 + r)

    repro_diff = np.array([abs(implied_traffic(med[50][c]) - implied_traffic(med37[c])) for c in topics])
    repro_ok_share = float(np.mean(repro_diff <= config.repro_tol))
    repro_spearman = float(spearmanr([med[50][c] for c in topics], [med37[c] for c in topics]).statistic)
    valid = repro_ok_share >= config.repro_share and repro_spearman >= config.repro_min_spearman
    LOGGER.info(
        f"k=50 reproduction: {repro_ok_share:.4f} within {config.repro_tol} neighbours; "
        f"max diff {repro_diff.max():.2f}; spearman {repro_spearman:.4f}"
    )

    def spearman(ids: list[int], k: int) -> float:
        s = [med[k][c] for c in ids]
        e = [e62[c]["refusal_effect"] for c in ids]
        return float(spearmanr(s, e).statistic) if len(ids) > 2 else float("nan")

    def auc(y: np.ndarray, s: np.ndarray) -> float | None:
        return float(roc_auc_score(y, s)) if len(set(y.tolist())) == 2 else None

    beh = [c for c in topics if c in e62]
    beh_random = [c for c in beh if e62[c]["arm"] == "random"]
    lab_ids = [c for c in topics if label.get(c) in ("real_gap", "partly", "covered")]
    y_rp = np.array([label[c] != "covered" for c in lab_ids]).astype(int)
    strict = [c for c in lab_ids if label[c] != "partly"]
    y_s = np.array([label[c] == "real_gap" for c in strict]).astype(int)
    eff = np.array([e62[c]["refusal_effect"] for c in beh])
    per_k = {}
    for k in KS:
        s_beh = np.array([med[k][c] for c in beh])
        per_k[k] = {
            "line": lines[k],
            "ceiling_share": ceiling[k],
            "topics_over_line": int(sum(med[k][c] >= lines[k] for c in beh)),
            "spearman_refusal_all": spearman(beh, k),
            "spearman_refusal_random": spearman(beh_random, k),
            "auc_over_refused": auc((eff >= config.over_refused).astype(int), s_beh),
            "auc_real_or_partly_vs_covered": auc(y_rp, np.array([med[k][c] for c in lab_ids])),
            "auc_real_vs_covered": auc(y_s, np.array([med[k][c] for c in strict])),
            "n_behaviour": len(beh),
            "n_random": len(beh_random),
            "n_labels": len(lab_ids),
        }

    # paired bootstrap over topics for k minus 50 in spearman with the refusal effect
    comparisons = {}
    s50 = np.array([med[50][c] for c in beh])
    for k in (100, 200):
        sk = np.array([med[k][c] for c in beh])
        diffs = []
        for _ in range(config.n_boot):
            idx = rng.integers(0, len(beh), len(beh))
            diffs.append(spearmanr(sk[idx], eff[idx]).statistic - spearmanr(s50[idx], eff[idx]).statistic)
        d = per_k[k]["spearman_refusal_all"] - per_k[50]["spearman_refusal_all"]
        lo, hi = np.nanquantile(diffs, [1 - config.ci_level, config.ci_level])
        comparisons[f"k{k}_minus_k50"] = {
            "difference": float(d),
            "ci": [float(lo), float(hi)],
            "clears": bool(valid and d >= config.margin and lo > 0),
        }

    verdict = (
        "invalid" if not valid else "larger_k_wins" if any(v["clears"] for v in comparisons.values()) else "k50_stays"
    )
    results = {
        "valid": valid,
        "k50_reproduction": {
            "share_within_one_neighbour": repro_ok_share,
            "max_neighbour_diff": float(repro_diff.max()),
            "spearman_with_exp37": repro_spearman,
        },
        "per_k": {str(k): v for k, v in per_k.items()},
        "comparisons": comparisons,
        "verdict": verdict,
        "n_topics": len(topics),
        "n_member_prompts": len(scored),
    }
    write_json(results, run / "decision.json")
    write_jsonl(
        [
            {
                "id": f"c{c}",
                "arm": e62[c]["arm"] if c in e62 else None,
                "refusal_effect": e62[c]["refusal_effect"] if c in e62 else None,
                "label": label.get(c),
                "size": len(members[c]),
                **{f"median_k{k}": round(med[k][c], 4) for k in KS},
                "median_exp37": round(med37[c], 4),
            }
            for c in topics
        ],
        run / "topics.jsonl",
    )
    print(f"Results saved to {run / 'decision.json'}")
    print(
        f"verdict {verdict}; k=50 reproduction {repro_ok_share:.4f} within one neighbour, spearman {repro_spearman:.4f}"
    )
    for k in KS:
        v = per_k[k]
        print(
            f"  k={k}: rho_all {v['spearman_refusal_all']:+.3f} rho_random {v['spearman_refusal_random']:+.3f} "
            f"auc_over_refused {v['auc_over_refused']} auc_labels {v['auc_real_or_partly_vs_covered']} "
            f"line {v['line']:.2f} over_line {v['topics_over_line']} ceiling {v['ceiling_share']:.3f}"
        )
    for name, v in comparisons.items():
        print(f"  {name}: {v['difference']:+.3f} CI [{v['ci'][0]:+.3f}, {v['ci'][1]:+.3f}] clears {v['clears']}")


if __name__ == "__main__":
    main()
