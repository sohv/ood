# stefan_detectors, CPU stage: turns the layer-16 features into eight detector scores, adds the tf-idf, auto-rater and random baselines, and tests all of them on behaviour (gap_overrefusal), the user's labels (precision_at_depth) and the planted check (density_ratio_needles).
# uv run -m scripts.run_detector_eval_stefan_detectors --run_dir results/raw/261027_stefan_detectors_detector_eval_v1 --features_dir results/raw/261027_stefan_detectors_detector_features_v1 --pipeline_million_dir results/raw/261021_pipeline_million_gap_pipeline_v2 --precision_at_depth_dir results/raw/261026_precision_at_depth_labels_v1 --misbehaviour_predictor_dir results/raw/261026_misbehaviour_predictor_behaviour_predictors_v1 --seed 0

import os

os.environ.setdefault("OMP_NUM_THREADS", "16")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "16")

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import simple_parsing
from scipy.stats import spearmanr
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.data.gap_pool import load_mix, load_traffic_pool
from src.data.io import load_jsonl
from src.metrics.gap_pipeline import text_hash
from src.metrics.io import write_json, write_jsonl
from src.utils.git import get_git_hash
from src.utils.logging import setup_logging
from src.utils.seed import set_seed

LOGGER = logging.getLogger(__name__)
DETECTORS = {
    "probe_reply": "eval_reply",
    "probe_prompt": "eval_prompt",
    "fvu_post_prompt": "eval_prompt",
    "delta_fvu_prompt": "eval_prompt",
    "fvu_post_reply": "eval_reply",
    "delta_fvu_reply": "eval_reply",
    "logprob_prompt": "eval_prompt",
    "logprob_reply": "eval_reply",
}
BASELINES = ("tfidf", "rater_B", "random")
REFERENCE = ("neighbour_count", "mix_share")


@dataclass
class EvalConfig:
    run_dir: str = ""
    features_dir: str = ""
    pipeline_million_dir: str = ""
    precision_at_depth_dir: str = ""
    misbehaviour_predictor_dir: str = ""
    auto_rater_finder_scores: str = "results/raw/261026_auto_rater_finder_autorater_v1/scores_flash_B_full.jsonl"
    n_tfidf: int = 200000
    n_boot: int = 2000
    alpha: float = 0.05
    over_refused: float = 0.20
    min_planted_auc: float = 0.90
    seed: int = 0
    num_tasks: int | None = None


def probe_scores(X: np.ndarray, y: np.ndarray, X_eval: np.ndarray, seed: int) -> tuple[np.ndarray, float]:
    # logistic probe pretraining (1) against post-training (0); returns P(pretraining) on eval rows and cross-fitted accuracy
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000))
    pred = cross_val_predict(clf, X, y, cv=5, method="predict_proba")[:, 1]
    acc = float(((pred > 0.5) == (y == 1)).mean())
    clf.fit(X, y)
    return clf.predict_proba(X_eval)[:, 1], acc


def tfidf_scores(config: EvalConfig, eval_texts: list[str], rng: np.random.Generator) -> np.ndarray:
    # count_vs_tfidf's word score: tf-idf unigrams and bigrams, SGD logistic regression traffic against mix; P(traffic) on the eval texts
    c37 = json.loads((Path(config.pipeline_million_dir) / "config.json").read_text())
    rows, _, _, _ = load_traffic_pool(
        c37["iterative_discovery_dir"],
        c37["self_nature_a_dir"],
        c37["emb_v2"],
        c37["new_lmsys_shards"],
        c37["embed_chars"],
    )
    _, _, m_text = load_mix(c37["sft_path"], c37["emb_v1"], with_text=True)
    eval_hashes = {text_hash(t) for t in eval_texts}
    t_pool = [r["norm"] for r in rows if text_hash(r["norm"]) not in eval_hashes]
    n = min(config.n_tfidf, len(t_pool), len(m_text))
    t_sample = [t_pool[i] for i in rng.choice(len(t_pool), size=n, replace=False)]
    m_sample = [m_text[i] for i in rng.choice(len(m_text), size=n, replace=False)]
    vec = TfidfVectorizer(ngram_range=(1, 2), max_features=200000, min_df=5, sublinear_tf=True)
    X = vec.fit_transform(t_sample + m_sample)
    y = np.array([1] * n + [0] * n)
    clf = SGDClassifier(loss="log_loss", class_weight="balanced", random_state=config.seed, max_iter=20, tol=1e-4)
    clf.fit(X, y)
    return clf.predict_proba(vec.transform(eval_texts))[:, 1]


def topic_medians(rows: list[dict], score: np.ndarray, task: str, set_name: str) -> dict[str, float]:
    groups: dict[str, list[float]] = {}
    for r, s in zip(rows, score, strict=True):
        if r["task"] == task and r["set"] == set_name:
            groups.setdefault(r["group"], []).append(float(s))
    # nan-safe: a record with no finite score (an empty raw text, a one-token reply) drops out of its topic's median
    return {g: float(np.nanmedian(v)) for g, v in groups.items() if np.isfinite(v).any()}


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    return float(spearmanr(a, b).correlation) if len(a) > 2 else float("nan")


def safe_auc(y, s) -> float:
    # nan when a class is missing, which happens in smoke runs
    y = np.asarray(y)
    return float(roc_auc_score(y, s)) if len(y) > 1 and 0 < y.sum() < len(y) else float("nan")


def main():
    config = simple_parsing.parse(EvalConfig)
    assert (
        config.run_dir
        and config.features_dir
        and config.pipeline_million_dir
        and config.precision_at_depth_dir
        and config.misbehaviour_predictor_dir
    )
    run = Path(config.run_dir)
    assert not run.exists(), f"{run} exists; results/raw is append-only"
    setup_logging(run)
    set_seed(config.seed)
    write_json({"git_hash": get_git_hash(), **asdict(config)}, run / "config.json")
    rng = np.random.default_rng(config.seed)

    feats = load_jsonl(Path(config.features_dir) / "features.jsonl")
    means = np.load(Path(config.features_dir) / "mean_acts.npy").astype(np.float32)
    assert len(feats) == len(means), (len(feats), len(means))
    by_set = {
        s: [i for i, r in enumerate(feats) if r["set"] == s]
        for s in ("eval_prompt", "eval_reply", "train_pre", "train_post", "train_post_prompt")
    }
    eval_prompt_rows = [feats[i] for i in by_set["eval_prompt"]]
    eval_reply_rows = [feats[i] for i in by_set["eval_reply"]]

    # detector scores per record, higher = more gap-like
    scores: dict[str, np.ndarray] = {}
    probe_acc = {}
    X_pre = means[by_set["train_pre"]]
    for name, post_set, eval_set in (
        ("probe_reply", "train_post", "eval_reply"),
        ("probe_prompt", "train_post_prompt", "eval_prompt"),
    ):
        X_post = means[by_set[post_set]]
        X = np.concatenate([X_pre, X_post])
        y = np.array([1] * len(X_pre) + [0] * len(X_post))
        scores[name], probe_acc[name] = probe_scores(X, y, means[by_set[eval_set]], config.seed)
        LOGGER.info(f"{name}: cross-fitted accuracy {probe_acc[name]:.3f}")
    for set_name, suffix in (("eval_prompt", "prompt"), ("eval_reply", "reply")):
        rows = [feats[i] for i in by_set[set_name]]
        # a per-record FVU needs a few tokens to have a variance; records with fewer than 3 content tokens get no FVU score
        short = np.array([r["n_content"] < 3 for r in rows])
        scores[f"fvu_post_{suffix}"] = np.where(short, np.nan, [r["fvu_post"] for r in rows])
        scores[f"delta_fvu_{suffix}"] = np.where(short, np.nan, [r["fvu_post"] - r["fvu_pre"] for r in rows])
        scores[f"logprob_{suffix}"] = np.array([r["lp_base"] - r["lp_dpo"] for r in rows])

    # baselines on the prompt records
    prompt_texts_path = Path(config.features_dir) / "eval_prompt_texts.jsonl"
    texts = {r["id"]: r["text"] for r in load_jsonl(prompt_texts_path)}
    eval_texts = [texts[r["id"]] for r in eval_prompt_rows]
    if config.num_tasks:
        config.n_tfidf = min(config.n_tfidf, 2000)
    scores["tfidf"] = tfidf_scores(config, eval_texts, rng)
    scores["random"] = rng.random(len(eval_prompt_rows))

    topics62 = {r["id"]: r for r in load_jsonl(Path(config.misbehaviour_predictor_dir) / "topics.jsonl")}
    effect = {c: r["refusal_effect"] for c, r in topics62.items()}

    # per-topic scores for the behaviour task
    per_topic: dict[str, dict[str, float]] = {}
    for name, set_name in DETECTORS.items():
        per_topic[name] = topic_medians(
            eval_prompt_rows if set_name == "eval_prompt" else eval_reply_rows, scores[name], "behaviour", set_name
        )
    per_topic["tfidf"] = topic_medians(eval_prompt_rows, scores["tfidf"], "behaviour", "eval_prompt")
    per_topic["random"] = topic_medians(eval_prompt_rows, scores["random"], "behaviour", "eval_prompt")
    per_topic["rater_B"] = {c: r["score_rater_B"] for c, r in topics62.items()}
    per_topic["neighbour_count"] = {c: r["score_neighbour_count"] for c, r in topics62.items()}
    per_topic["mix_share"] = {c: r["score_mix_share"] for c, r in topics62.items()}
    common = sorted(set.intersection(*(set(v) for v in per_topic.values())) & set(effect))
    LOGGER.info(f"{len(common)} topics with every score")
    eff = np.array([effect[c] for c in common])
    over = (eff >= config.over_refused).astype(int)
    arms = np.array([topics62[c]["arm"] for c in common])

    t1 = {}
    for name in list(DETECTORS) + list(BASELINES) + list(REFERENCE):
        s = np.array([per_topic[name][c] for c in common])
        t1[name] = {
            "spearman_refusal": spearman(s, eff),
            "spearman_random_arm": spearman(s[arms == "random"], eff[arms == "random"]),
            "auc_over_refused": float(roc_auc_score(over, s)) if 0 < over.sum() < len(over) else float("nan"),
        }
    best_baseline = max(BASELINES, key=lambda b: t1[b]["spearman_refusal"])
    level = config.alpha / len(DETECTORS)
    boot_rng = np.random.default_rng(config.seed)
    idx_boot = [boot_rng.choice(len(common), size=len(common), replace=True) for _ in range(config.n_boot)]
    verdicts = {}
    for name in DETECTORS:
        s = np.array([per_topic[name][c] for c in common])
        b = np.array([per_topic[best_baseline][c] for c in common])
        diffs = np.array([spearman(s[i], eff[i]) - spearman(b[i], eff[i]) for i in idx_boot])
        diffs = diffs[~np.isnan(diffs)]
        lo, hi = (
            (float(np.quantile(diffs, level / 2)), float(np.quantile(diffs, 1 - level / 2)))
            if len(diffs)
            else (float("nan"), float("nan"))
        )
        beats_all = all(t1[name]["spearman_refusal"] > t1[bl]["spearman_refusal"] for bl in BASELINES)
        verdicts[name] = {
            "best_baseline": best_baseline,
            "diff_vs_best": t1[name]["spearman_refusal"] - t1[best_baseline]["spearman_refusal"],
            "ci": [lo, hi],
            "beats_baselines": bool(beats_all and lo > 0),
        }

    # T2: the user's precision_at_depth second-round labels, real gap or partly covered against covered
    e61 = json.loads((Path(config.precision_at_depth_dir) / "decision.json").read_text())
    labels = {r["id"]: r["label"] for r in e61["new_rows"]} | {r["id"]: r["second"] for r in e61["relabel_rows"]}
    t2 = {}
    for name in [n for n, s in DETECTORS.items() if s == "eval_prompt"] + ["tfidf", "random"]:
        med = topic_medians(eval_prompt_rows, scores[name], "labels", "eval_prompt")
        ids = [c for c in med if labels.get(c) in ("real_gap", "partly", "covered")]
        y_len = np.array([labels[c] != "covered" for c in ids]).astype(int)
        y_str = np.array([labels[c] == "real_gap" for c in ids if labels[c] != "partly"]).astype(int)
        s_all = np.array([med[c] for c in ids])
        s_str = np.array([med[c] for c in ids if labels[c] != "partly"])
        t2[name] = {
            "n": len(ids),
            "auc_real_or_partly_vs_covered": safe_auc(y_len, s_all),
            "auc_real_vs_covered": safe_auc(y_str, s_str),
        }
    # the auto-rater scored the loop's ranked topics in auto_rater_finder; its label AUC is on the labelled topics it covers
    rater56 = {r["id"]: r["gap_score"] for r in load_jsonl(config.auto_rater_finder_scores)}
    ids = [c for c in labels if c in rater56 and labels[c] in ("real_gap", "partly", "covered")]
    if len(ids) > 2:
        y_len = np.array([labels[c] != "covered" for c in ids]).astype(int)
        t2["rater_B"] = {
            "n": len(ids),
            "auc_real_or_partly_vs_covered": safe_auc(y_len, [rater56[c] for c in ids]),
            "auc_real_vs_covered": float("nan"),
        }

    # T3: planted legal and medical against traffic, per record
    t3 = {}
    planted_rows = [(i, r) for i, r in enumerate(eval_prompt_rows) if r["task"] == "planted"]
    y3 = np.array([r["group"] in ("legal", "medical") for _, r in planted_rows]).astype(int)
    for name in [n for n, s in DETECTORS.items() if s == "eval_prompt"] + ["tfidf", "random"]:
        s3 = np.array([scores[name][i] for i, _ in planted_rows])
        ok = np.isfinite(s3)
        t3[name] = safe_auc(y3[ok], s3[ok])
    for name in DETECTORS:
        verdicts[name]["planted_ok"] = (
            bool(t3[name] >= config.min_planted_auc) if name in t3 and not np.isnan(t3[name]) else None
        )
        verdicts[name]["verdict"] = (
            "not_checked_on_planted"
            if verdicts[name]["planted_ok"] is None
            else "broken"
            if not verdicts[name]["planted_ok"]
            else "beats"
            if verdicts[name]["beats_baselines"]
            else "does_not_beat"
        )

    nan_counts = {name: int((~np.isfinite(s)).sum()) for name, s in scores.items()}
    LOGGER.info(f"non-finite record scores {nan_counts}")
    decision = {
        "n_topics": len(common),
        "nan_record_scores": nan_counts,
        "t1_behaviour": t1,
        "t1_verdicts": verdicts,
        "t2_labels": t2,
        "t3_planted_auroc": t3,
        "probe_cross_fitted_accuracy": probe_acc,
        "best_baseline": best_baseline,
        "ci_level": 1 - level,
        "git_hash": get_git_hash(),
    }
    out = write_json(decision, run / "decision.json")
    write_jsonl(
        [
            {
                "id": c,
                "arm": topics62[c]["arm"],
                "refusal_effect": effect[c],
                **{f"score_{n}": per_topic[n][c] for n in per_topic},
            }
            for c in common
        ],
        run / "topics.jsonl",
    )
    write_jsonl(
        [
            {
                "id": r["id"],
                "set": r["set"],
                "group": r["group"],
                "task": r["task"],
                **{n: float(scores[n][i]) for n, s in DETECTORS.items() if s == "eval_prompt"},
                "tfidf": float(scores["tfidf"][i]),
            }
            for i, r in enumerate(eval_prompt_rows)
        ],
        run / "prompt_scores.jsonl",
    )

    print("T1 behaviour, Spearman with the refusal effect (all topics, random arm), AUC over-refused")
    for name, v in t1.items():
        print(f"  {name}: {v['spearman_refusal']:.3f} {v['spearman_random_arm']:.3f} {v['auc_over_refused']:.3f}")
    print(f"best baseline {best_baseline}")
    for name, v in verdicts.items():
        print(
            f"  {name}: diff {v['diff_vs_best']:+.3f} ci {v['ci'][0]:+.3f} to {v['ci'][1]:+.3f}; planted {t3.get(name, float('nan')):.3f}; {v['verdict']}"
        )
    print("T2 labels AUC (real or partly vs covered; real vs covered)")
    for name, v in t2.items():
        print(f"  {name}: {v['auc_real_or_partly_vs_covered']:.3f} {v['auc_real_vs_covered']:.3f} (n {v['n']})")
    print(f"Results saved to {out}")


if __name__ == "__main__":
    main()
