# exp18 analysis: judge validity, noise budget, and whether prompt OOD scores separate red from green prompts.
# uv run -m scripts.run_sharechat_analysis --run_dir results/raw/261019_exp18_sharechat_v1 --pools_dir data/processed/exp18 --exp01_pools_dir data/processed/exp01 --seed 0

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import simple_parsing
from huggingface_hub import hf_hub_download
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from src.data.io import load_jsonl
from src.interp.features import LAYERS
from src.metrics.behaviour import ORANGE_AT_OR_BELOW, PRINCIPLES, cohen_kappa, icc1
from src.metrics.detectors import auroc, score_cell, stratified_auroc, tfidf_mix_vs_dolma
from src.metrics.io import write_json, write_jsonl
from src.utils.git import get_git_hash
from src.utils.logging import setup_logging
from src.utils.seed import set_seed

LOGGER = logging.getLogger(__name__)


@dataclass
class SharechatAnalysisConfig:
    run_dir: str = ""
    pools_dir: str = ""
    exp01_pools_dir: str = ""
    seed: int = 0
    layer: int = 16
    pooling: str = "cmean"
    n_rollouts: int = 6
    n_boot: int = 1000


def main():
    config = simple_parsing.parse(SharechatAnalysisConfig)
    assert (
        config.run_dir and config.pools_dir and config.exp01_pools_dir
    ), "run_dir, pools_dir, exp01_pools_dir required"
    run, pools = Path(config.run_dir), Path(config.pools_dir)
    setup_logging(run)
    set_seed(config.seed)
    write_json({"git_hash": get_git_hash(), **asdict(config)}, run / "analysis_config.json")
    rng = np.random.default_rng(config.seed)

    pool = load_jsonl(pools / "sharechat.jsonl")
    rollouts = load_jsonl(run / "rollouts.jsonl")
    grades = {g["id"]: g for g in load_jsonl(run / "grades_r0.jsonl")}
    assert len(rollouts) == len(pool) * config.n_rollouts == len(grades), "row counts differ"
    assert {r["id"] for r in rollouts} == set(grades), "grade ids differ from rollout ids"

    # judge validity on mood replies
    validation = {r["id"]: r for r in load_jsonl(pools / "judge_validation.jsonl")}
    vgrades = load_jsonl(run / "grades_validation.jsonl")
    assert {g["id"] for g in vgrades} == set(validation), "validation grades incomplete"
    validity = {}
    for group in ("harmful", "benign"):
        gs = [g for g in vgrades if validation[g["id"]]["group"] == group]
        graded = [g["orange"] for g in gs if g["orange"] is not None]
        validity[group] = {"n": len(gs), "n_null": len(gs) - len(graded), "orange_rate": float(np.mean(graded))}
    validity["per_domain_orange"] = {
        d: float(
            np.mean([g["orange"] for g in vgrades if validation[g["id"]]["domain"] == d and g["orange"] is not None])
        )
        for d in sorted({v["domain"] for v in validation.values()})
    }
    diff = validity["harmful"]["orange_rate"] - validity["benign"]["orange_rate"]
    is_judge_valid = validity["harmful"]["orange_rate"] >= 0.50 and diff >= 0.30
    validity |= {"harmful_minus_benign": diff, "valid": is_judge_valid}

    # judge test-retest on 300 replies graded twice
    retest = load_jsonl(run / "grades_r1_retest.jsonl")
    pairs = [(grades[g["id"]], g) for g in retest if grades[g["id"]]["scores"] is not None and g["scores"] is not None]
    retest_stats = {
        "n": len(retest),
        "n_pairs_both_graded": len(pairs),
        "orange_agreement": float(np.mean([a["orange"] == b["orange"] for a, b in pairs])),
        "orange_kappa": cohen_kappa(
            np.array([a["orange"] for a, _ in pairs]), np.array([b["orange"] for _, b in pairs])
        ),
        "principle_exact_agreement": {
            p: float(np.mean([a["scores"][p] == b["scores"][p] for a, b in pairs])) for p in PRINCIPLES
        },
    }

    # per-prompt outcomes
    by_prompt: dict[str, list] = {r["id"]: [None] * config.n_rollouts for r in pool}
    reply_len: dict[str, list] = {r["id"]: [] for r in pool}
    for r in rollouts:
        by_prompt[r["prompt_id"]][r["rollout"]] = grades[r["id"]]["orange"]
        reply_len[r["prompt_id"]].append(r["n_tokens"])
    full = [pid for pid, v in by_prompt.items() if None not in v]
    matrix = np.array([by_prompt[pid] for pid in full], dtype=float)
    red = [pid for pid in full if all(by_prompt[pid])]
    green = [pid for pid in full if not any(by_prompt[pid])]
    graded = [g for g in grades.values() if g["scores"] is not None]
    noise = {
        "icc1_orange": icc1(matrix),
        "n_prompts_all_graded": len(full),
        "n_red": len(red),
        "n_green": len(green),
        "orange_rate_per_reply": float(np.mean([g["orange"] for g in graded])),
        "n_null": len(grades) - len(graded),
        "null_reasons": {
            k: sum(g["null_reason"] == k for g in grades.values())
            for k in {g["null_reason"] for g in grades.values()}
            if k
        },
        "cap_share": float(np.mean([r["finish_reason"] == "length" for r in rollouts])),
        "per_principle_low_rate": {
            p: float(np.mean([g["scores"][p] <= ORANGE_AT_OR_BELOW for g in graded])) for p in PRINCIPLES
        },
        "orange_rate_by_platform": {
            plat: float(
                np.mean([v for r in pool if r["platform"] == plat for v in by_prompt[r["id"]] if v is not None])
            )
            for plat in sorted({r["platform"] for r in pool})
        },
        "judge_cost_usd": float(sum(g["cost"] or 0 for g in grades.values())),
    }

    # features and the pipeline check against exp01's stored idfit features
    feats = {
        n: np.load(run / "extract" / "features" / f"{n}.npz")
        for n in ("idfit", "sharechat", "dolma_full", "dolma_matched")
    }
    assert feats["sharechat"]["ids"].tolist() == [r["id"] for r in pool], "feature rows misaligned with pool"
    li = feats["idfit"]["layers"].tolist().index(config.layer)
    x = {n: f[config.pooling][:, li].astype(np.float32) for n, f in feats.items()}
    stored = np.load(
        hf_hub_download("detectood/ood", "results/raw/261018_exp01_repro_v1/features/idfit.npz", repo_type="dataset")
    )
    # exp01 stored no layers field; it used the module default LAYERS
    assert stored[config.pooling].shape[1] == len(LAYERS), "unexpected layer count in exp01 features"
    stored_x = stored[config.pooling][:, LAYERS.index(config.layer)].astype(np.float32)
    # exp01 ran on other hardware and truncated at 1,024 tokens, so compare same-length rows and report the drift
    same_len = stored["n_content_tokens"] == feats["idfit"]["n_content_tokens"]
    a, b = stored_x[same_len], x["idfit"][same_len]
    rel = np.linalg.norm(a - b, axis=1) / np.linalg.norm(a, axis=1)
    pipeline_check = {
        "ids_match": stored["ids"].tolist() == feats["idfit"]["ids"].tolist(),
        "rows_same_length": int(same_len.sum()),
        "rows_longer_than_exp01_truncation": int((~same_len).sum()),
        "same_length_rel_err_median": float(np.median(rel)),
        "same_length_rel_err_p99": float(np.percentile(rel, 99)),
        "same_length_cosine_min": float(((a * b).sum(1) / np.linalg.norm(a, axis=1) / np.linalg.norm(b, axis=1)).min()),
    }
    assert pipeline_check["ids_match"], "re-extracted idfit ids differ from exp01's"
    scores = score_cell(x, ("sharechat",))["sharechat"]
    texts = [r["text"] for r in pool]
    tfidf = tfidf_mix_vs_dolma(
        [r["text"] for r in load_jsonl(Path(config.exp01_pools_dir) / "idfit.jsonl")],
        [r["text"] for r in load_jsonl(Path(config.exp01_pools_dir) / "dolma.jsonl")],
        config.seed,
    )
    scores["tfidf"] = tfidf(texts)
    prompt_len = feats["sharechat"]["n_content_tokens"].astype(float)
    assert prompt_len.tolist() == [float(r["n_content_tokens"]) for r in pool], "content token counts differ"
    scores["prompt_length"] = prompt_len
    scores["reply_length_mean"] = np.array([np.mean(reply_len[r["id"]]) for r in pool])
    index = {r["id"]: i for i, r in enumerate(pool)}
    ri, gi = np.array([index[p] for p in red]), np.array([index[p] for p in green])

    detectors = ("dknn_full", "knn", "maha", "dmaha_full", "tfidf", "prompt_length", "reply_length_mean")
    q = {}
    for det in detectors:
        s = scores[det]
        st = stratified_auroc(s[ri], s[gi], prompt_len[ri], prompt_len[gi])
        q[det] = {
            "auroc_raw": auroc(s[ri], s[gi]),
            "auroc_stratified": st["auroc"],
            "n_bands": st["n_bands"],
            "coverage": st["coverage"],
        }

    def strat(det: str, a: np.ndarray, b: np.ndarray) -> float | None:
        return stratified_auroc(scores[det][a], scores[det][b], prompt_len[a], prompt_len[b])["auroc"]

    boots, diffs = [], []
    for _ in range(config.n_boot):
        a, b = rng.choice(ri, len(ri)), rng.choice(gi, len(gi))
        d, t = strat("dknn_full", a, b), strat("tfidf", a, b)
        if d is not None:
            boots.append(d)
        if d is not None and t is not None:
            diffs.append(d - t)
    q["dknn_full"]["ci_stratified"] = (
        [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]
        if len(boots) >= 0.9 * config.n_boot
        else None
    )
    diff_ci = (
        [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))]
        if len(diffs) >= 0.9 * config.n_boot
        else None
    )

    probe_auroc = None
    if min(len(ri), len(gi)) >= 10:
        xy = np.vstack([x["sharechat"][ri], x["sharechat"][gi]])
        y = np.r_[np.ones(len(ri)), np.zeros(len(gi))]
        oof = np.zeros(len(y))
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=config.seed).split(xy, y):
            sc = StandardScaler().fit(xy[tr])
            oof[te] = (
                LogisticRegression(C=0.1, max_iter=5000, random_state=config.seed)
                .fit(sc.transform(xy[tr]), y[tr])
                .predict_proba(sc.transform(xy[te]))[:, 1]
            )
        probe_auroc = auroc(oof[y == 1], oof[y == 0])

    d_strat, t_strat = q["dknn_full"]["auroc_stratified"], q["tfidf"]["auroc_stratified"]
    len_raw = q["prompt_length"]["auroc_raw"]
    if len(red) < 30 or len(green) < 30 or d_strat is None or t_strat is None or diff_ci is None:
        verdict = "unmeasurable"
    elif d_strat >= 0.65 and d_strat >= len_raw + 0.05 and d_strat - t_strat >= 0.05 and diff_ci[0] > 0:
        verdict = "predicts"
    elif d_strat <= 0.60 or d_strat <= max(len_raw, t_strat):
        verdict = "does_not"
    else:
        verdict = "ambiguous"
    if not is_judge_valid:
        verdict = f"no_verdict_judge_invalid ({verdict})"

    near = [g for g in graded if min(g["scores"].values()) in (2, 3)]
    near_pick = [near[i] for i in rng.choice(len(near), size=min(20, len(near)), replace=False)]
    users = {r["id"]: r["text"] for r in pool}
    replies = {r["id"]: r["reply"] for r in rollouts}
    write_jsonl(
        [
            {
                "id": g["id"],
                "min_score": min(g["scores"].values()),
                "orange": g["orange"],
                "scores": g["scores"],
                "judge": g["raw"],
                "user": users[g["prompt_id"]],
                "reply": replies[g["id"]],
            }
            for g in near_pick
        ],
        run / "near_threshold_sample.jsonl",
    )
    examples = [rollouts[i] for i in rng.choice(len(rollouts), size=5, replace=False)]
    decision = {
        "judge_validity": validity,
        "retest": retest_stats,
        "noise": noise,
        "pipeline_check": pipeline_check,
        "red_vs_green": q,
        "dknn_minus_tfidf_stratified": None if d_strat is None or t_strat is None else d_strat - t_strat,
        "dknn_minus_tfidf_ci": diff_ci,
        "probe_cv_auroc": probe_auroc,
        "verdict": verdict,
        "examples": [
            {
                "id": e["id"],
                "user": users[e["prompt_id"]][:500],
                "reply": e["reply"][:800],
                "finish_reason": e["finish_reason"],
                "grade": grades[e["id"]]["scores"],
            }
            for e in examples
        ],
    }
    write_json(decision, run / "decision.json")
    red_set, green_set = set(red), set(green)
    write_jsonl(
        [
            {
                "id": r["id"],
                "platform": r["platform"],
                "n_graded": sum(v is not None for v in by_prompt[r["id"]]),
                "n_orange": sum(bool(v) for v in by_prompt[r["id"]]),
                "is_red": r["id"] in red_set,
                "is_green": r["id"] in green_set,
                **{d: float(scores[d][i]) for d in detectors},
            }
            for i, r in enumerate(pool)
        ],
        run / "prompt_scores.jsonl",
    )
    print(
        json.dumps(
            {
                "validity": validity,
                "noise": {
                    k: noise[k]
                    for k in ("icc1_orange", "n_red", "n_green", "orange_rate_per_reply", "n_null", "cap_share")
                },
                "verdict": verdict,
            },
            indent=1,
        )
    )
    print(f"decision written to {run / 'decision.json'}")


if __name__ == "__main__":
    main()
