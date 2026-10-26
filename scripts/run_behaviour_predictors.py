# exp62: which saved score predicts exp52's per-topic over-refusal: the neighbour count, the retrieval auto-rater, nearest similarity, mix share, or dumb baselines.
# uv run -m scripts.run_behaviour_predictors --run_dir results/raw/261026_exp62_behaviour_predictors_v1 --exp37_dir results/raw/261021_exp37_gap_pipeline_v2 --exp52_grades results/raw/261023_exp52_refusal_v2/grades.jsonl --exp56_dir results/raw/261026_exp56_autorater_v1 --exp58_dir results/raw/261026_exp58_similarity_gt_v1 --exp55_dir results/raw/261026_exp55_eval_set_v1 --seed 0

import json
import logging
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import simple_parsing
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from src.data.io import load_jsonl
from src.metrics.io import write_json, write_jsonl
from src.utils.git import get_git_hash
from src.utils.logging import setup_logging
from src.utils.seed import set_seed

LOGGER = logging.getLogger(__name__)
SCORES = ("neighbour_count", "rater_B", "rater_A", "nearest_similarity", "mix_share", "cluster_size", "random")


@dataclass
class BehaviourPredictorsConfig:
    run_dir: str = ""
    exp37_dir: str = ""
    exp52_grades: str = ""
    exp56_dir: str = ""
    exp58_dir: str = ""
    exp55_dir: str = ""
    seed: int = 0
    over_refused: float = 0.20
    top_k: int = 50
    n_boot: int = 2000
    margin: float = 0.10
    floor: float = 0.20


def main():
    config = simple_parsing.parse(BehaviourPredictorsConfig)
    assert all(
        [config.run_dir, config.exp37_dir, config.exp52_grades, config.exp56_dir, config.exp58_dir, config.exp55_dir]
    )
    run = Path(config.run_dir)
    assert not run.exists(), f"{run} exists; results/raw is append-only"
    setup_logging(run)
    set_seed(config.seed)
    write_json({"git_hash": get_git_hash(), **asdict(config)}, run / "config.json")
    rng = np.random.default_rng(config.seed)

    # behaviour per topic from exp52: refusal and harmless refusal, members against their matched controls
    counts = defaultdict(lambda: {"member": [0, 0, 0], "control": [0, 0, 0], "arm": None})
    for g in load_jsonl(config.exp52_grades):
        if g.get("grade") is None:
            continue
        c = int(g["group"].split("_c")[1])
        side = counts[c][g["role"]]
        side[0] += 1
        side[1] += int(g["refused"])
        side[2] += int(g["refused"] and g["grade"].get("request_harmful") == "no")
        counts[c]["arm"] = g["arm"]
    topics = sorted(counts)
    behaviour = {}
    for c in topics:
        m, k = counts[c]["member"], counts[c]["control"]
        assert m[0] > 0 and k[0] > 0, (c, m, k)
        behaviour[c] = {
            "arm": counts[c]["arm"],
            "n_member": m[0],
            "n_control": k[0],
            "refusal_effect": m[1] / m[0] - k[1] / k[0],
            "harmless_refusal_effect": m[2] / m[0] - k[2] / k[0],
        }
    LOGGER.info(
        f"{len(topics)} topics: {sum(b['arm'] == 'gap' for b in behaviour.values())} gap, {sum(b['arm'] == 'random' for b in behaviour.values())} random"
    )

    clusters = {int(r["id"][1:]): r for r in load_jsonl(Path(config.exp37_dir) / "clusters.jsonl")}
    n_ranked = sum(1 for r in clusters.values() if r["rank"])
    rater = {
        v: {
            int(r["id"][1:]): r["gap_score"]
            for r in load_jsonl(Path(config.exp56_dir) / f"scores_flash_{v}_full.jsonl")
        }
        for v in ("A", "B")
    }
    sim = {int(r["id"][1:]): r["sim_med"] for r in load_jsonl(Path(config.exp58_dir) / "topic_similarity.jsonl")}
    share = {
        int(r["id"][1:]): r["mix_share"] for r in load_jsonl(Path(config.exp55_dir) / "cluster_ground_truth.jsonl")
    }
    scores = {
        "neighbour_count": {c: float(n_ranked - clusters[c]["rank"]) for c in topics},
        "rater_B": {c: rater["B"][c] for c in topics},
        "rater_A": {c: rater["A"][c] for c in topics},
        "nearest_similarity": {c: 1 - sim[c] for c in topics},
        "mix_share": {c: 1 - share[c] for c in topics},
        "cluster_size": {c: float(clusters[c]["size"]) for c in topics},
        "random": dict(zip(topics, rng.random(len(topics)))),
    }

    def evaluate(subset: list[int]) -> dict:
        out = {}
        a = np.array([behaviour[c]["refusal_effect"] for c in subset])
        b = np.array([behaviour[c]["harmless_refusal_effect"] for c in subset])
        over = (a >= config.over_refused).astype(int)
        for name in SCORES:
            s = np.array([scores[name][c] for c in subset])
            top = np.argsort(-s, kind="stable")[: config.top_k]
            out[name] = {
                "spearman_refusal": float(spearmanr(s, a).statistic),
                "spearman_harmless_refusal": float(spearmanr(s, b).statistic),
                "auc_over_refused": float(roc_auc_score(over, s)) if 0 < over.sum() < len(over) else None,
                "precision_at_top": float(over[top].mean()),
            }
        out["_n"] = len(subset)
        out["_over_refused"] = int(over.sum())
        return out

    subsets = {
        "all": topics,
        "random_arm": [c for c in topics if behaviour[c]["arm"] == "random"],
        "gap_arm": [c for c in topics if behaviour[c]["arm"] == "gap"],
    }
    results = {name: evaluate(sub) for name, sub in subsets.items()}

    # paired bootstrap over topics for the count-minus-rater difference in spearman with the refusal effect
    a_all = np.array([behaviour[c]["refusal_effect"] for c in topics])
    s_count = np.array([scores["neighbour_count"][c] for c in topics])
    s_rater = np.array([scores["rater_B"][c] for c in topics])
    diffs = []
    for _ in range(config.n_boot):
        idx = rng.integers(0, len(topics), len(topics))
        diffs.append(spearmanr(s_count[idx], a_all[idx]).statistic - spearmanr(s_rater[idx], a_all[idx]).statistic)
    lo, hi = float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))
    d = results["all"]["neighbour_count"]["spearman_refusal"] - results["all"]["rater_B"]["spearman_refusal"]
    best = max(results["all"]["neighbour_count"]["spearman_refusal"], results["all"]["rater_B"]["spearman_refusal"])
    if best < config.floor:
        verdict = "neither_predicts_behaviour"
    elif d >= config.margin and lo > 0:
        verdict = "count_beats_rater"
    elif -d >= config.margin and hi < 0:
        verdict = "rater_beats_count"
    else:
        verdict = "equivalent"

    write_jsonl(
        [
            {"id": f"c{c}", **behaviour[c], **{f"score_{n}": round(float(scores[n][c]), 4) for n in SCORES}}
            for c in topics
        ],
        run / "topics.jsonl",
    )
    decision = {
        "git_hash": get_git_hash(),
        "verdict": verdict,
        "count_minus_rater_spearman_refusal": {"difference": d, "ci95": [lo, hi]},
        "results": results,
        "sanity": {
            "n_topics": len(topics),
            "grades_used": int(sum(v["n_member"] + v["n_control"] for v in behaviour.values())),
        },
    }
    write_json(decision, run / "decision.json")
    print(f"verdict {verdict}; count minus rater (spearman with refusal effect) {d:.3f} CI [{lo:.3f}, {hi:.3f}]")
    for sub, r in results.items():
        print(f"  {sub} (n {r['_n']}, over-refused {r['_over_refused']}):")
        for name in SCORES:
            v = r[name]
            auc = f"{v['auc_over_refused']:.3f}" if v["auc_over_refused"] is not None else "na"
            print(
                f"    {name:20s} rho_refusal {v['spearman_refusal']:+.3f} rho_harmless {v['spearman_harmless_refusal']:+.3f} auc {auc} p@{config.top_k} {v['precision_at_top']:.2f}"
            )
    print(f"decision written to {run / 'decision.json'}")


if __name__ == "__main__":
    main()
