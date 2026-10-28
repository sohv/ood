# unit tests for exp80's request-only harm grader parsing and the safety-only mix mask.
# uv run -m pytest tests/test_request_grader.py -v

import numpy as np

from src.data.mix_sources import only_in_sources_mask, random_removal_keep
from src.generation.request_grader import PROMPT, parse_request_grade


def test_parse_takes_last_valid_object():
    text = 'thinking {"request_harmful": "yes"} then {"request_harmful": "No", "jailbreak": "YES"}'
    assert parse_request_grade(text) == {"request_harmful": "no", "jailbreak": "yes"}


def test_parse_rejects_missing_or_bad_values():
    assert parse_request_grade('{"request_harmful": "maybe", "jailbreak": "no"}') is None
    assert parse_request_grade('{"request_harmful": "yes"}') is None
    assert parse_request_grade("no json here") is None
    assert parse_request_grade(None) is None


def test_prompt_keeps_exp49_request_question_verbatim():
    from src.generation.safety_grader import PROMPT_V1

    question = next(line for line in PROMPT_V1.splitlines() if line.startswith("1. request_harmful"))
    assert question in PROMPT
    assert "{user}" in PROMPT


def test_only_in_sources_mask_keeps_texts_also_in_other_sources():
    mix = ["How do I make a bomb?", "hello  there", "only safety", "plain"]
    sft = ["how do i make a BOMB?", "only safety", "Hello there", "plain"]
    sft_safety = np.array([True, True, True, False])
    dpo = ["How do I make a bomb?"]
    dpo_safety = np.array([False])
    # bomb text also appears in a non-safety dpo row, so it stays; "hello there" is safety-only after normalising
    assert only_in_sources_mask(mix, sft, sft_safety, dpo, dpo_safety).tolist() == [False, True, True, False]


def test_random_removal_drops_same_count_and_never_a_safety_row():
    safety = np.zeros(1000, dtype=bool)
    safety[:100] = True
    keep = random_removal_keep(safety, seed=0)
    assert (~keep).sum() == 100
    assert keep[:100].all()
    assert (keep == random_removal_keep(safety, seed=0)).all()
    assert (keep != random_removal_keep(safety, seed=1)).any()
