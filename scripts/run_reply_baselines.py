# reply_length_kill_test: per-reply auroc of sharechat_reply_quality's orange labels from reply length and out-of-fold reply tf-idf (stage-classifier kill test).
# uv run -m scripts.run_reply_baselines --sharechat_reply_quality_dir results/raw/261019_sharechat_reply_quality_sharechat_v1 --output_dir results/raw/261020_reply_length_kill_test_reply_baselines_v1 --seed 0

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import simple_parsing
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

from src.data.io import load_jsonl
from src.metrics.detectors import auroc, stratified_auroc
from src.metrics.io import write_json, write_jsonl
from src.utils.git import get_git_hash
from src.utils.logging import setup_logging
from src.utils.seed import set_seed

LOGGER = logging.getLogger(__name__)

REPLY_BANDS = (1, 16, 32, 64, 128, 256, 512, 1024, 2048, 2049)
KILL_AT = 0.78


@dataclass
class ReplyBaselinesConfig:
    sharechat_reply_quality_dir: str = ""
    output_dir: str = ""
    model_id: str = "allenai/Olmo-3-7B-Instruct-DPO"
    seed: int = 0
    n_folds: int = 5
    c: float = 1.0
    n_boot: int = 1000


def tfidf_model(c: float, seed: int) -> tuple[TfidfVectorizer, LogisticRegression]:
    return TfidfVectorizer(sublinear_tf=True, min_df=2, max_features=200_000), LogisticRegression(
        C=c, max_iter=5000, random_state=seed
    )


def out_of_fold_tfidf(texts: list[str], y: np.ndarray, groups: np.ndarray, config: ReplyBaselinesConfig) -> np.ndarray:
    scores = np.zeros(len(texts))
    for train, test in GroupKFold(n_splits=config.n_folds).split(texts, y, groups):
        vectorizer, clf = tfidf_model(config.c, config.seed)
        clf.fit(vectorizer.fit_transform([texts[i] for i in train]), y[train])
        scores[test] = clf.predict_proba(vectorizer.transform([texts[i] for i in test]))[:, 1]
    return scores


def message_bootstrap_ci(score: np.ndarray, y: np.ndarray, groups: np.ndarray, n_boot: int, seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    uniq, inverse = np.unique(groups, return_inverse=True)
    rows_of = np.split(np.argsort(inverse, kind="stable"), np.cumsum(np.bincount(inverse))[:-1])
    values = []
    for _ in range(n_boot):
        idx = np.concatenate([rows_of[m] for m in rng.integers(0, len(uniq), len(uniq))])
        values.append(auroc(score[idx][y[idx] == 1], score[idx][y[idx] == 0]))
    return [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]


def main():
    config = simple_parsing.parse(ReplyBaselinesConfig)
    assert (
        config.sharechat_reply_quality_dir and config.output_dir
    ), "sharechat_reply_quality_dir and output_dir required"
    src, out = Path(config.sharechat_reply_quality_dir), Path(config.output_dir)
    setup_logging(out)
    set_seed(config.seed)
    write_json({"git_hash": get_git_hash(), **asdict(config)}, out / "config.json")

    rollouts = load_jsonl(src / "rollouts.jsonl")
    grades = {g["id"]: g for g in load_jsonl(src / "grades_r0.jsonl")}
    messages = {m["id"]: m for m in load_jsonl(src / "prompt_scores.jsonl")}
    assert len(rollouts) == len(grades) == 24_000, "row counts differ"
    assert {r["id"] for r in rollouts} == set(grades), "grade ids differ from rollout ids"
    rows = [r for r in rollouts if grades[r["id"]]["orange"] is not None]
    n_null = len(rollouts) - len(rows)
    LOGGER.info(f"{len(rows)=} {n_null=}")

    y = np.array([grades[r["id"]]["orange"] for r in rows], dtype=int)
    groups = np.array([r["prompt_id"] for r in rows])
    texts = [r["reply"] for r in rows]
    reply_len = np.array([r["n_tokens"] for r in rows], dtype=float)
    is_capped = np.array([r["finish_reason"] == "length" for r in rows])
    prompt_len = np.array([messages[r["prompt_id"]]["prompt_length"] for r in rows], dtype=float)

    reply_tfidf = out_of_fold_tfidf(texts, y, groups, config)
    # negative control: each message takes another message's labels, so label clustering by message is kept
    rng = np.random.default_rng(config.seed)
    uniq, inverse = np.unique(groups, return_inverse=True)
    labels_of = [y[inverse == i] for i in range(len(uniq))]
    donor = rng.permutation(len(uniq))
    shuffled_y = np.zeros_like(y)
    for i in range(len(uniq)):
        shuffled_y[inverse == i] = np.resize(labels_of[donor[i]], (inverse == i).sum())
    shuffled_tfidf = out_of_fold_tfidf(texts, shuffled_y, groups, config)

    scores = {"reply_len": reply_len, "reply_tfidf": reply_tfidf, "prompt_len": prompt_len}
    per_reply = {
        name: {
            "auroc": auroc(s[y == 1], s[y == 0]),
            "ci": message_bootstrap_ci(s, y, groups, config.n_boot, config.seed),
        }
        for name, s in scores.items()
    }
    per_reply["shuffled_tfidf_control"] = {
        "auroc": auroc(shuffled_tfidf[shuffled_y == 1], shuffled_tfidf[shuffled_y == 0])
    }

    # capped replies (2,048 tokens) land in their own top band
    band_len = np.where(is_capped, 2048.5, reply_len)
    tfidf_within_len = stratified_auroc(
        reply_tfidf[y == 1], reply_tfidf[y == 0], band_len[y == 1], band_len[y == 0], bands=REPLY_BANDS
    )

    # message level, like for like with sharechat_reply_quality's 0.83 probe
    is_red = np.array([messages[m]["is_red"] for m in uniq])
    is_green = np.array([messages[m]["is_green"] for m in uniq])
    mean_len = np.array([reply_len[inverse == i].mean() for i in range(len(uniq))])
    mean_tfidf = np.array([reply_tfidf[inverse == i].mean() for i in range(len(uniq))])
    sharechat_reply_quality_mean_len = np.array([messages[m]["reply_length_mean"] for m in uniq])
    message_level = {
        "n_red": int(is_red.sum()),
        "n_green": int(is_green.sum()),
        "reply_len_mean": auroc(mean_len[is_red], mean_len[is_green]),
        "reply_tfidf_mean": auroc(mean_tfidf[is_red], mean_tfidf[is_green]),
        "sharechat_reply_quality_reply_length_mean_reproduced": auroc(
            sharechat_reply_quality_mean_len[is_red], sharechat_reply_quality_mean_len[is_green]
        ),
        "max_abs_diff_mean_len_vs_sharechat_reply_quality": float(
            np.abs(mean_len - sharechat_reply_quality_mean_len)[is_red | is_green].max()
        ),
    }

    vectorizer, clf = tfidf_model(config.c, config.seed)
    clf.fit(vectorizer.fit_transform(texts), y)
    vocab, coef = vectorizer.get_feature_names_out(), clf.coef_[0]
    order = np.argsort(coef)
    top_words = {
        "orange": [[vocab[i], float(coef[i])] for i in order[::-1][:20]],
        "not_orange": [[vocab[i], float(coef[i])] for i in order[:20]],
    }

    orange_idx = rng.choice(np.where(y == 1)[0], 5, replace=False)
    examples = [
        {
            "id": rows[i]["id"],
            "reply_len": int(reply_len[i]),
            "reply_tfidf": float(reply_tfidf[i]),
            "judge_reason": json.loads(grades[rows[i]["id"]]["raw"])["reason"],
            "reply": texts[i][:1500],
        }
        for i in orange_idx
    ]

    point, upper = per_reply["reply_len"]["auroc"], per_reply["reply_len"]["ci"][1]
    verdict = "drop" if point >= KILL_AT else ("go" if upper < KILL_AT else "ambiguous")
    decision = {
        "n_rows": len(rollouts),
        "n_graded": len(rows),
        "n_null": n_null,
        "n_orange": int(y.sum()),
        "orange_rate": float(y.mean()),
        "per_reply": per_reply,
        "tfidf_within_reply_length": tfidf_within_len,
        "capped": {
            "n": int(is_capped.sum()),
            "orange_rate_capped": float(y[is_capped].mean()),
            "orange_rate_not_capped": float(y[~is_capped].mean()),
        },
        "message_level": message_level,
        "top_words": top_words,
        "examples": examples,
        "kill_at": KILL_AT,
        "verdict": verdict,
    }
    write_json(decision, out / "decision.json")
    write_jsonl(
        [
            {
                "id": r["id"],
                "prompt_id": r["prompt_id"],
                "orange": int(y[i]),
                "reply_len": int(reply_len[i]),
                "is_capped": bool(is_capped[i]),
                "reply_tfidf": float(reply_tfidf[i]),
                "prompt_len": float(prompt_len[i]),
            }
            for i, r in enumerate(rows)
        ],
        out / "reply_scores.jsonl",
    )
    print(f"Results saved to {out / 'decision.json'} and {out / 'reply_scores.jsonl'}")
    print(f"per reply: {json.dumps({k: round(v['auroc'], 4) for k, v in per_reply.items()})}")
    print(f"verdict: {verdict}")


if __name__ == "__main__":
    main()
