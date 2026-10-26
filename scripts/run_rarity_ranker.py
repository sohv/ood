# exp63: a combined label-free rarity ranker for per-topic over-refusal, against the count, the mix share and the auto-rater, plus the planted test.
# uv run -m scripts.run_rarity_ranker --run_dir results/raw/261026_exp63_rarity_ranker_v1 --exp37_dir results/raw/261021_exp37_gap_pipeline_v2 --exp55_dir results/raw/261026_exp55_eval_set_v1 --exp58_dir results/raw/261026_exp58_similarity_gt_v1 --exp62_dir results/raw/261026_exp62_behaviour_predictors_v1 --exp57_dir results/raw/261026_exp57_rater_confirm_v1 --seed 0

import json
import logging
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import simple_parsing
import torch
from scipy.stats import rankdata, spearmanr
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold

from src.data.gap_pool import load_mix, load_traffic_pool
from src.data.io import load_jsonl
from src.metrics.gap_pipeline import density_ratios_batched, text_hash
from src.metrics.io import write_json, write_jsonl
from src.utils.git import get_git_hash
from src.utils.logging import setup_logging
from src.utils.seed import set_seed

LOGGER = logging.getLogger(__name__)
FEATURES = ("count_rank", "mix_rarity", "sim_rarity", "sim_low_share", "cluster_size", "template_share", "median_chars")


@dataclass
class RarityRankerConfig:
    run_dir: str = ""
    exp37_dir: str = ""
    exp55_dir: str = ""
    exp58_dir: str = ""
    exp62_dir: str = ""
    exp57_dir: str = ""
    seed: int = 0
    alpha: float = 1.0
    over_refused: float = 0.20
    k: int = 50
    top_n: int = 293
    template_chars: int = 60
    n_boot: int = 2000
    margin: float = 0.10
    planted_ok: int = 6
    planted_fail: int = 5
    ratio_multiple: float = 3.0
    n_random_subsets: int = 1000
    subset_size: int = 500
    folds: int = 5
    hi_cos: float = 0.80


def mix_share(labels: np.ndarray, m_np: np.ndarray, t_np: np.ndarray, keep: np.ndarray, n_clusters: int) -> np.ndarray:
    # every kept mix prompt goes to its nearest traffic centroid; share = mix rate / (mix rate + traffic rate)
    cent = np.zeros((n_clusters, t_np.shape[1]), dtype=np.float32)
    np.add.at(cent, labels, t_np.astype(np.float32))
    cent /= np.linalg.norm(cent, axis=1, keepdims=True) + 1e-12
    cent_gpu = torch.from_numpy(cent).cuda()
    kept = np.where(keep)[0]
    assign = np.empty(len(kept), dtype=np.int64)
    for s in range(0, len(kept), 65536):
        q = torch.from_numpy(m_np[kept[s : s + 65536]]).cuda().float()
        assign[s : s + 65536] = (q @ cent_gpu.T).argmax(1).cpu().numpy()
    rate_t = np.bincount(labels, minlength=n_clusters) / len(labels)
    rate_m = np.bincount(assign, minlength=n_clusters) / len(kept)
    return rate_m / (rate_m + rate_t + 1e-12)


def main():
    config = simple_parsing.parse(RarityRankerConfig)
    assert all(
        [config.run_dir, config.exp37_dir, config.exp55_dir, config.exp58_dir, config.exp62_dir, config.exp57_dir]
    )
    run = Path(config.run_dir)
    assert not run.exists(), f"{run} exists; results/raw is append-only"
    setup_logging(run)
    set_seed(config.seed)
    write_json({"git_hash": get_git_hash(), **asdict(config)}, run / "config.json")
    rng = np.random.default_rng(config.seed)

    c37 = json.loads((Path(config.exp37_dir) / "config.json").read_text())
    rows, t_np, _, _ = load_traffic_pool(
        c37["exp09_dir"], c37["exp31_dir"], c37["emb_v2"], c37["new_lmsys_shards"], c37["embed_chars"]
    )
    t_hash = np.array([text_hash(r["norm"]) for r in rows], dtype=np.int64)
    labels = np.load(Path(config.exp37_dir) / "labels.npy")
    m_np, m_hash = load_mix(c37["sft_path"], c37["emb_v1"])
    clusters = {int(c["id"][1:]): c for c in load_jsonl(Path(config.exp37_dir) / "clusters.jsonl")}
    ranked = sorted((c for c in clusters if clusters[c]["rank"]), key=lambda c: clusters[c]["rank"])
    n_clusters = int(labels.max()) + 1
    members = {c: np.where(labels == c)[0] for c in ranked}

    # label-free features for every ranked topic, from saved runs plus two text statistics
    share55 = {
        int(r["id"][1:]): r["mix_share"] for r in load_jsonl(Path(config.exp55_dir) / "cluster_ground_truth.jsonl")
    }
    sim58 = {int(r["id"][1:]): r for r in load_jsonl(Path(config.exp58_dir) / "topic_similarity.jsonl")}
    feats = {}
    for c in ranked:
        norms = [rows[i]["norm"] for i in members[c]]
        _, top = Counter(n[: config.template_chars] for n in norms).most_common(1)[0]
        feats[c] = {
            "count_rank": float(len(ranked) - clusters[c]["rank"]),
            "mix_rarity": 1 - share55[c],
            "sim_rarity": 1 - sim58[c]["sim_med"],
            "sim_low_share": 1 - sim58[c]["share_hi"],
            "cluster_size": float(len(members[c])),
            "template_share": top / len(members[c]),
            "median_chars": float(np.median([len(rows[i]["text"]) for i in members[c]])),
        }
    behaviour = {int(r["id"][1:]): r for r in load_jsonl(Path(config.exp62_dir) / "topics.jsonl")}
    gap_arm = [c for c in behaviour if behaviour[c]["arm"] == "gap"]
    random_arm = [c for c in behaviour if behaviour[c]["arm"] == "random"]
    assert all(c in feats for c in behaviour), "a behaviour topic is not a ranked cluster"

    def matrix(topics: list[int]) -> np.ndarray:
        return np.array([[feats[c][f] for f in FEATURES] for c in topics])

    def rank_average(topics: list[int], keys: tuple[str, ...], values: dict[int, dict] | None = None) -> np.ndarray:
        src = values or feats
        return np.mean([rankdata([src[c][k] for c in topics]) for k in keys], axis=0)

    x_train, y_train = matrix(gap_arm), np.array([behaviour[c]["refusal_effect"] for c in gap_arm])
    mu, sd = x_train.mean(0), x_train.std(0) + 1e-9
    r2 = Ridge(alpha=config.alpha).fit((x_train - mu) / sd, y_train)
    pair = [FEATURES.index("count_rank"), FEATURES.index("mix_rarity")]
    r3 = Ridge(alpha=config.alpha).fit(((x_train - mu) / sd)[:, pair], y_train)
    cv = []
    for tr, te in KFold(config.folds, shuffle=True, random_state=config.seed).split(x_train):
        m = Ridge(alpha=config.alpha).fit(((x_train[tr] - mu) / sd), y_train[tr])
        cv.append(float(spearmanr(m.predict((x_train[te] - mu) / sd), y_train[te]).statistic))

    y_test = np.array([behaviour[c]["refusal_effect"] for c in random_arm])
    over = (y_test >= config.over_refused).astype(int)
    x_test = (matrix(random_arm) - mu) / sd
    scores = {
        "R1_rank_average": rank_average(random_arm, ("count_rank", "mix_rarity")),
        "R2_ridge_all": r2.predict(x_test),
        "R3_ridge_count_mix": r3.predict(x_test[:, pair]),
        "count": np.array([feats[c]["count_rank"] for c in random_arm]),
        "mix_share": np.array([feats[c]["mix_rarity"] for c in random_arm]),
        "rater_B": np.array([behaviour[c]["score_rater_B"] for c in random_arm]),
    }
    test = {}
    for name, s in scores.items():
        test[name] = {
            "spearman": float(spearmanr(s, y_test).statistic),
            "auc_over_refused": float(roc_auc_score(over, s)) if 0 < over.sum() < len(over) else None,
        }
    best_single = max(("count", "mix_share", "rater_B"), key=lambda n: test[n]["spearman"])
    boot = {}
    for name in ("R1_rank_average", "R2_ridge_all", "R3_ridge_count_mix"):
        d = []
        for _ in range(config.n_boot):
            idx = rng.integers(0, len(random_arm), len(random_arm))
            d.append(
                spearmanr(scores[name][idx], y_test[idx]).statistic
                - spearmanr(scores[best_single][idx], y_test[idx]).statistic
            )
        boot[name] = {
            "difference": test[name]["spearman"] - test[best_single]["spearman"],
            "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
        }
    LOGGER.info(f"random arm: {test}; best single {best_single}")

    # planted test: exp57's planted topics, their mix neighbours deleted, count ratio and mix share recomputed
    d57 = json.loads((Path(config.exp57_dir) / "decision.json").read_text())
    planted = d57["sanity"]["planted"]
    m_gpu = torch.from_numpy(m_np).cuda()
    keep = np.ones(len(m_np), dtype=bool)
    for c in planted:
        q = torch.from_numpy(t_np[members[c]]).cuda()
        idx = torch.cat(
            [(q[s : s + 1024] @ m_gpu.T).topk(config.k, dim=1).indices.flatten() for s in range(0, len(q), 1024)]
        )
        keep[np.unique(idx.cpu().numpy())] = False
    kept = np.where(keep)[0]
    ratio_abl = density_ratios_batched(
        t_np,
        t_hash,
        np.arange(len(rows)),
        torch.from_numpy(t_np).cuda(),
        torch.from_numpy(t_hash).cuda(),
        m_gpu[torch.from_numpy(kept).cuda()],
        torch.from_numpy(m_hash[kept]).cuda(),
        k=config.k,
    )
    c_ratio = {c: float(np.median(ratio_abl[members[c]])) for c in ranked}
    sub_rng = np.random.default_rng(config.seed)
    line = config.ratio_multiple * float(
        np.median(
            [
                np.median(ratio_abl[sub_rng.choice(len(rows), size=config.subset_size, replace=False)])
                for _ in range(config.n_random_subsets)
            ]
        )
    )
    share_abl = mix_share(labels, m_np, t_np, keep, n_clusters)
    abl = {c: {"count_rank": 0.0, "mix_rarity": 1 - float(share_abl[c])} for c in ranked}
    order_count = sorted(ranked, key=lambda c: (-c_ratio[c], c))
    for pos, c in enumerate(order_count):
        abl[c]["count_rank"] = float(len(ranked) - pos)
    planted_over = sum(c_ratio[c] >= line for c in planted)
    r1_abl = rank_average(ranked, ("count_rank", "mix_rarity"), abl)
    x_abl = (np.array([[abl[c]["count_rank"], abl[c]["mix_rarity"]] for c in ranked]) - mu[pair]) / sd[pair]
    r3_abl = r3.predict(x_abl)
    planted_found = {}
    for name, s in (
        ("R1_rank_average", r1_abl),
        ("R3_ridge_count_mix", r3_abl),
        ("count", np.array([abl[c]["count_rank"] for c in ranked])),
        ("mix_share", np.array([abl[c]["mix_rarity"] for c in ranked])),
    ):
        top = {ranked[i] for i in np.argsort(-s, kind="stable")[: config.top_n]}
        planted_found[name] = int(sum(c in top for c in planted))

    verdicts = {}
    for name in ("R1_rank_average", "R2_ridge_all", "R3_ridge_count_mix"):
        b = boot[name]
        pf = planted_found.get(name)
        if b["difference"] >= config.margin and b["ci95"][0] > 0 and (pf is None or pf >= config.planted_ok):
            verdicts[name] = "better"
        elif b["difference"] <= -config.margin or (pf is not None and pf <= config.planted_fail):
            verdicts[name] = "worse"
        else:
            verdicts[name] = "equivalent"

    write_jsonl(
        [
            {
                "id": f"c{c}",
                **feats[c],
                "refusal_effect": behaviour[c]["refusal_effect"] if c in behaviour else None,
                "arm": behaviour[c]["arm"] if c in behaviour else None,
                "ratio_ablated": round(c_ratio[c], 3),
                "mix_share_ablated": round(float(share_abl[c]), 4),
            }
            for c in ranked
        ],
        run / "topic_features.jsonl",
    )
    decision = {
        "git_hash": get_git_hash(),
        "verdicts": verdicts,
        "best_single_score_on_random_arm": best_single,
        "random_arm_test": test,
        "bootstrap_vs_best_single": boot,
        "cross_validation_gap_arm_spearman": {"folds": cv, "mean": float(np.mean(cv))},
        "ridge_coefficients": {
            "R2": dict(zip(FEATURES, [float(v) for v in r2.coef_])),
            "R3": {"count_rank": float(r3.coef_[0]), "mix_rarity": float(r3.coef_[1])},
        },
        "planted": {
            "topics": planted,
            "over_ablated_line": planted_over,
            "ablated_line": line,
            "found_in_top_293": planted_found,
        },
        "sanity": {
            "n_train": len(gap_arm),
            "n_test": len(random_arm),
            "n_ranked": len(ranked),
            "over_refused_in_test": int(over.sum()),
            "mix_rows_removed": int((~keep).sum()),
        },
    }
    write_json(decision, run / "decision.json")
    print(f"verdicts {verdicts}; best single on random arm: {best_single}")
    for name, v in test.items():
        extra = f" vs best {boot[name]['difference']:+.3f} CI {boot[name]['ci95']}" if name in boot else ""
        print(f"  {name:20s} spearman {v['spearman']:+.3f} auc {v['auc_over_refused']}{extra}")
    print(f"  cross-validation on the 181 gap topics: {np.mean(cv):.3f} (folds {[round(x, 2) for x in cv]})")
    print(f"  planted over ablated line {planted_over}/10 (line {line:.2f}); found in top 293: {planted_found}")
    print(f"  ridge R2 coefficients {decision['ridge_coefficients']['R2']}")
    print(f"decision written to {run / 'decision.json'}")


if __name__ == "__main__":
    main()
