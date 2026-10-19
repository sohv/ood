import json

import numpy as np
import pandas as pd

from src.data.gaps import draw_controls, merge_groups
from src.data.sharechat import clean
from src.metrics.behaviour import PRINCIPLES, cohen_kappa, holm, icc1, is_orange, parse_judge, rate_ratio

FULL = {k: 5 for k in PRINCIPLES}


def judge_text(scores: dict) -> str:
    return json.dumps({"reason": "fine", "scores": scores})


def test_parse_judge_accepts_full_json_and_code_fences():
    assert parse_judge(judge_text(FULL)) == FULL
    assert parse_judge("```json\n" + judge_text(FULL) + "\n```") == FULL


def test_parse_judge_rejects_missing_extra_or_bad_values():
    missing = {k: 5 for k in PRINCIPLES[:-1]}
    assert parse_judge(judge_text(missing)) is None
    assert parse_judge(judge_text(FULL | {"extra": 5})) is None
    assert parse_judge(judge_text(FULL | {"safety": 0})) is None
    assert parse_judge(judge_text(FULL | {"safety": 6})) is None
    assert parse_judge(judge_text(FULL | {"safety": "5"})) is None
    assert parse_judge(judge_text(FULL | {"safety": 4.5})) is None
    assert parse_judge(judge_text(FULL | {"safety": True})) is None
    assert parse_judge("") is None
    assert parse_judge(None) is None
    assert parse_judge("no json here") is None


def test_orange_rule_is_any_principle_at_or_below_two():
    assert is_orange(FULL) is False
    assert is_orange(FULL | {"verbosity": 3}) is False
    assert is_orange(FULL | {"verbosity": 2}) is True
    assert is_orange(FULL | {"safety": 1}) is True
    assert is_orange(None) is None


def test_icc_extremes():
    # every prompt fully consistent across rollouts: icc 1
    perfect = np.array([[1] * 6, [0] * 6, [1] * 6, [0] * 6], dtype=float)
    assert abs(icc1(perfect) - 1.0) < 1e-9
    # identical prompts, all variation within: icc at or below 0
    within = np.array([[1, 0, 1, 0, 1, 0]] * 4, dtype=float)
    assert icc1(within) <= 0


def test_kappa_extremes():
    a = np.array([1, 0, 1, 0, 1, 1, 0, 0])
    assert abs(cohen_kappa(a, a) - 1.0) < 1e-9
    assert cohen_kappa(a, 1 - a) < 0


def test_sharechat_clean_filters_and_dedupes():
    df = pd.DataFrame(
        {
            "plain_text": ["Hello there", "hello   THERE", "", None, "my name is <REDACTED>", "see <URL>", "ok"],
            "platform": ["chatgpt"] * 7,
        }
    )
    out, counts = clean(df)
    assert out["text"].tolist() == ["Hello there", "ok"]
    assert counts == {
        "first_user_turns": 7,
        "dropped_empty": 2,
        "dropped_redaction": 2,
        "dropped_duplicate": 1,
        "clean": 2,
    }


def test_merge_groups_links_close_centroids_only():
    emb = np.array([[1.0, 0.0], [0.99, 0.141], [0.0, 1.0], [0.0, 1.0]])
    emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
    members = {"a": np.array([0]), "b": np.array([1]), "c": np.array([2, 3])}
    groups = merge_groups(members, ["a", "b", "c"], emb, 0.85)
    assert sorted(map(sorted, groups)) == [["a", "b"], ["c"]]


def test_draw_controls_matches_source_and_band_without_replacement():
    source = np.array(["w", "w", "w", "w", "l", "l", "w", "w"])
    band = np.array([1, 1, 1, 1, 1, 1, 2, 2])
    eligible = np.array([False, False, True, True, True, True, True, True])
    rng = np.random.default_rng(0)
    out = draw_controls(np.array([0, 1]), source, band, eligible, 1, rng)
    assert set(out.ravel()) == {2, 3}
    assert not eligible[[2, 3]].any()


def test_holm_matches_hand_computation():
    adjusted = holm({"a": 0.01, "b": 0.04, "c": 0.03})
    assert adjusted == {"a": 0.03, "c": 0.06, "b": 0.06}


def test_rate_ratio_edge_cases():
    assert rate_ratio(np.array([0.5, 0.5]), np.array([0.25, 0.25])) == 2.0
    assert rate_ratio(np.array([0.5]), np.array([0.0])) == float("inf")
    assert np.isnan(rate_ratio(np.array([0.0]), np.array([0.0])))
