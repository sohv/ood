# dpo_looks_like_sft: sft-vs-dpo reply classifiers (tf-idf and activations), scored on dpo replies as a predictor of sharechat_reply_quality's orange labels.
# uv run -m scripts.run_stage_classifier --run_dir results/raw/261020_dpo_looks_like_sft_stage_classifier_v1 --sft_rollouts results/raw/261020_dpo_looks_like_sft_rollouts_v1/rollouts.jsonl --sharechat_reply_quality_dir results/raw/261019_sharechat_reply_quality_sharechat_v1 --reply_length_kill_test_dir results/raw/261020_reply_length_kill_test_reply_baselines_v1 --seed 0

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import simple_parsing
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from src.data.io import load_jsonl
from src.metrics.detectors import auroc, stratified_auroc
from src.metrics.io import write_json, write_jsonl
from src.utils.git import get_git_hash
from src.utils.logging import setup_logging
from src.utils.seed import set_seed

LOGGER = logging.getLogger(__name__)

REPLY_BANDS = (1, 16, 32, 64, 128, 256, 512, 1024, 2048, 2049)
WORDS_PER_REPLY = 0.7248
WORDS_WITHIN_LENGTH = 0.6802
VALID_AT = 0.75
DISCARD_AT = 0.60


@dataclass
class StageClassifierConfig:
    run_dir: str = ""
    sft_rollouts: str = ""
    sharechat_reply_quality_dir: str = ""
    reply_length_kill_test_dir: str = ""
    model_id: str = "allenai/Olmo-3-7B-Instruct-DPO"
    seed: int = 0
    n_folds: int = 5
    c_text: float = 1.0
    c_act: float = 0.1
    n_boot: int = 1000


def fit_stage(name: str, x, y: np.ndarray, config: StageClassifierConfig):
    # returns a function giving p(sft) for new inputs
    if name == "text_stage":
        transform = TfidfVectorizer(sublinear_tf=True, min_df=2, max_features=200_000)
        c = config.c_text
    else:
        transform = StandardScaler()
        c = config.c_act
    clf = LogisticRegression(C=c, max_iter=5000, random_state=config.seed).fit(transform.fit_transform(x), y)
    return lambda z: clf.predict_proba(transform.transform(z))[:, 1]


def message_rows(groups: np.ndarray) -> list[np.ndarray]:
    _, inverse = np.unique(groups, return_inverse=True)
    return np.split(np.argsort(inverse, kind="stable"), np.cumsum(np.bincount(inverse))[:-1])


def paired_bootstrap(a: np.ndarray, b: np.ndarray, y: np.ndarray, groups: np.ndarray, n_boot: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    rows_of = message_rows(groups)
    va, diffs = [], []
    for _ in range(n_boot):
        idx = np.concatenate([rows_of[m] for m in rng.integers(0, len(rows_of), len(rows_of))])
        yy = y[idx]
        x, z = auroc(a[idx][yy == 1], a[idx][yy == 0]), auroc(b[idx][yy == 1], b[idx][yy == 0])
        va.append(x)
        diffs.append(x - z)
    return {
        "ci95": [float(np.percentile(va, 2.5)), float(np.percentile(va, 97.5))],
        "diff_ci975": [float(np.percentile(diffs, 1.25)), float(np.percentile(diffs, 98.75))],
    }


def main():
    config = simple_parsing.parse(StageClassifierConfig)
    assert (
        config.run_dir
        and config.sft_rollouts
        and config.sharechat_reply_quality_dir
        and config.reply_length_kill_test_dir
    ), "paths required"
    run = Path(config.run_dir)
    setup_logging(run)
    set_seed(config.seed)
    write_json({"git_hash": get_git_hash(), **asdict(config)}, run / "config.json")

    dpo = {r["id"]: r for r in load_jsonl(Path(config.sharechat_reply_quality_dir) / "rollouts.jsonl")}
    sft = {r["id"]: r for r in load_jsonl(config.sft_rollouts)}
    graded = load_jsonl(Path(config.reply_length_kill_test_dir) / "reply_scores.jsonl")
    feats = np.load(run / "reply_features.npz")
    assert len(sft) == 8000 and all(r["rollout"] in (0, 1) for r in sft.values()), "expected 2 sft replies x 4000"
    n_empty_sft = sum(1 for r in sft.values() if not r["reply"].strip())
    assert {g["id"] for g in graded} <= set(dpo), "reply_length_kill_test ids not in sharechat_reply_quality rollouts"
    assert (feats["n_content_tokens"] > 0).all(), "a reply has no content tokens"
    row_of = {(int(s), i): k for k, (s, i) in enumerate(zip(feats["stage"], feats["ids"]))}
    assert len(row_of) == len(dpo) + len(sft) - n_empty_sft, "feature rows do not cover every non-empty reply"

    # the classifier pool: every non-empty sft reply and dpo replies 0 and 1; evaluation: every graded dpo reply
    train_rows = [(1, i, sft[i]["prompt_id"]) for i in sft if (1, i) in row_of] + [
        (0, i, r["prompt_id"]) for i, r in dpo.items() if r["rollout"] in (0, 1)
    ]
    eval_ids = [g["id"] for g in graded]
    y = np.array([g["orange"] for g in graded])
    groups = np.array([g["prompt_id"] for g in graded])
    reply_len = np.array([g["reply_len"] for g in graded], dtype=float)
    band_len = np.array([2048.5 if g["is_capped"] else g["reply_len"] for g in graded], dtype=float)
    reply_tfidf = np.array([g["reply_tfidf"] for g in graded])

    def inputs(name: str, pairs: list[tuple[int, str]]):
        if name == "text_stage":
            return [(sft if s else dpo)[i]["reply"] for s, i in pairs]
        return feats["cmean"][[row_of[p] for p in pairs]].astype(np.float32)

    names = ("text_stage", "act_stage")
    messages = np.array(sorted({p for _, _, p in train_rows}))
    eval_scores = {name: np.zeros(len(eval_ids)) for name in names}
    control = {name: ([], []) for name in names}
    for train_msg_idx, test_msg_idx in GroupKFold(n_splits=config.n_folds).split(messages, groups=messages):
        train_msgs, test_msgs = set(messages[train_msg_idx]), set(messages[test_msg_idx])
        tr = [(s, i) for s, i, p in train_rows if p in train_msgs]
        te = [(s, i) for s, i, p in train_rows if p in test_msgs]
        ev = [k for k, p in enumerate(groups) if p in test_msgs]
        for name in names:
            predict = fit_stage(name, inputs(name, tr), np.array([s for s, _ in tr]), config)
            control[name][0].extend(predict(inputs(name, te)))
            control[name][1].extend(s for s, _ in te)
            eval_scores[name][ev] = predict(inputs(name, [(0, eval_ids[k]) for k in ev]))
        LOGGER.info(f"fold done: {len(tr)} train, {len(te)} held-out stage rows, {len(ev)} graded dpo replies")

    results = {}
    for name, score in eval_scores.items():
        c_score, c_stage = np.array(control[name][0]), np.array(control[name][1])
        stage_auroc = auroc(c_score[c_stage == 1], c_score[c_stage == 0])
        boot = paired_bootstrap(score, reply_tfidf, y, groups, config.n_boot, config.seed)
        within = stratified_auroc(score[y == 1], score[y == 0], band_len[y == 1], band_len[y == 0], bands=REPLY_BANDS)
        per_reply = auroc(score[y == 1], score[y == 0])
        results[name] = {
            "heldout_sft_vs_dpo_auroc": stage_auroc,
            "is_valid": stage_auroc >= VALID_AT,
            "orange_auroc_per_reply": per_reply,
            "ci95": boot["ci95"],
            "minus_reply_tfidf_ci975": boot["diff_ci975"],
            "orange_auroc_within_length": within["auroc"],
            "within_length_bands": within["bands"],
            "is_cleared": stage_auroc >= VALID_AT
            and per_reply > WORDS_PER_REPLY
            and boot["diff_ci975"][0] > 0
            and within["auroc"] is not None
            and within["auroc"] > WORDS_WITHIN_LENGTH,
        }
    valid = [n for n in results if results[n]["is_valid"]]
    if any(results[n]["is_cleared"] for n in valid):
        verdict = "cleared"
    elif not valid:
        verdict = "invalid"
    elif all(results[n]["orange_auroc_per_reply"] <= DISCARD_AT for n in valid):
        verdict = "discard"
    else:
        verdict = "ambiguous"

    act_minus_text = paired_bootstrap(
        eval_scores["act_stage"], eval_scores["text_stage"], y, groups, config.n_boot, config.seed
    )

    # reply length alone as a stage classifier, for reference against the positive-control gate
    stage_len = np.array([(sft if s else dpo)[i]["n_tokens"] for s, i, _ in train_rows], dtype=float)
    stage_lab = np.array([s for s, _, _ in train_rows])
    msgs = {m["id"]: m for m in load_jsonl(Path(config.sharechat_reply_quality_dir) / "prompt_scores.jsonl")}
    rows_of = message_rows(groups)
    msg_ids = [groups[r[0]] for r in rows_of]
    red = np.array([msgs[m]["is_red"] for m in msg_ids])
    green = np.array([msgs[m]["is_green"] for m in msg_ids])
    message_level = {}
    for name, score in eval_scores.items():
        mean = np.array([score[r].mean() for r in rows_of])
        message_level[name] = auroc(mean[red], mean[green])

    vectorizer = TfidfVectorizer(sublinear_tf=True, min_df=2, max_features=200_000)
    clf = LogisticRegression(C=config.c_text, max_iter=5000, random_state=config.seed)
    clf.fit(vectorizer.fit_transform(inputs("text_stage", [(s, i) for s, i, _ in train_rows])), stage_lab)
    vocab, order = vectorizer.get_feature_names_out(), np.argsort(clf.coef_[0])
    top_words = {"sft": [vocab[i] for i in order[::-1][:20]], "dpo": [vocab[i] for i in order[:20]]}

    example_msg = min(sft.values(), key=lambda r: r["id"])["prompt_id"]
    example = {
        "prompt_id": example_msg,
        "sft_reply": sft[f"{example_msg}_r0"]["reply"][:1200],
        "dpo_reply": dpo[f"{example_msg}_r0"]["reply"][:1200],
    }

    decision = {
        "n_sft": len(sft),
        "n_empty_sft": n_empty_sft,
        "n_graded_dpo": len(graded),
        "n_orange": int(y.sum()),
        "results": results,
        "act_minus_text_per_reply": auroc(eval_scores["act_stage"][y == 1], eval_scores["act_stage"][y == 0])
        - auroc(eval_scores["text_stage"][y == 1], eval_scores["text_stage"][y == 0]),
        "act_minus_text_ci975": act_minus_text["diff_ci975"],
        "reply_len_as_stage_classifier": auroc(stage_len[stage_lab == 1], stage_len[stage_lab == 0]),
        "median_reply_tokens": {
            "sft": float(np.median([r["n_tokens"] for r in sft.values()])),
            "dpo": float(np.median([r["n_tokens"] for r in dpo.values()])),
        },
        "capped_share": {
            "sft": float(np.mean([r["finish_reason"] == "length" for r in sft.values()])),
            "dpo": float(np.mean([r["finish_reason"] == "length" for r in dpo.values()])),
        },
        "message_level_red_vs_green": message_level,
        "top_words": top_words,
        "example": example,
        "baselines_from_reply_length_kill_test": {
            "reply_len": 0.5405,
            "prompt_len": 0.6485,
            "reply_tfidf": WORDS_PER_REPLY,
        },
        "verdict": verdict,
    }
    write_json(decision, run / "decision.json")
    write_jsonl(
        [
            {
                "id": i,
                "orange": int(y[k]),
                **{n: float(eval_scores[n][k]) for n in eval_scores},
                "reply_tfidf": float(reply_tfidf[k]),
            }
            for k, i in enumerate(eval_ids)
        ],
        run / "stage_scores.jsonl",
    )
    print(f"Results saved to {run / 'decision.json'} and {run / 'stage_scores.jsonl'}")
    print(
        json.dumps(
            {
                n: {
                    k: results[n][k]
                    for k in ("heldout_sft_vs_dpo_auroc", "orange_auroc_per_reply", "orange_auroc_within_length")
                }
                for n in results
            }
        )
    )
    print(f"verdict: {verdict}")


if __name__ == "__main__":
    main()
