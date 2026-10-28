# unit tests for exp87's confirmation question: prompt wording and reply parsing.
# uv run -m pytest tests/test_confirmation.py -v

from src.metrics.confirmation import CONFIRM_PROMPT, confirmed_by, parse_confirmation
from src.metrics.discovery import naming_prompt


def test_parse_confirmation_reads_every_field():
    out = parse_confirmation(
        "NAME: Spanish language messages\nDESCRIPTION: Mixed requests in Spanish.\nSHARED: 6\n"
        "LANGUAGE: Spanish\nNOT_ENGLISH: 22\nFORMAT: NONE\nIN_FORMAT: 0"
    )
    assert out["name"] == "Spanish language messages"
    assert out["shared"] == 6 and out["not_english"] == 22 and out["in_format"] == 0
    assert out["language"] == "Spanish" and out["format"] is None


def test_parse_confirmation_handles_missing_and_broken_replies():
    assert parse_confirmation(None)["shared"] is None and parse_confirmation(None)["not_english"] is None
    out = parse_confirmation(
        "NAME: x\nDESCRIPTION: y\nSHARED: many\nLANGUAGE: NONE\nNOT_ENGLISH: 0\nFORMAT: JSON\nIN_FORMAT: 14"
    )
    assert out["shared"] is None and out["language"] is None and out["format"] == "JSON" and out["in_format"] == 14


def test_confirmed_by_names_the_criteria_that_reach_the_cut():
    assert confirmed_by({"shared": 13, "not_english": 2, "in_format": None}, 13) == ["topic"]
    assert confirmed_by({"shared": 5, "not_english": 20, "in_format": 15}, 13) == ["language", "format"]
    assert confirmed_by({"shared": None, "not_english": None, "in_format": None}, 13) == []


def test_confirm_prompt_keeps_the_grain_wording_and_asks_the_new_counts():
    prompt = naming_prompt(["hola", "bonjour"], n=25, seed=0, template=CONFIRM_PROMPT)
    assert "broad field or a single language" in prompt
    for key in ("SHARED:", "LANGUAGE:", "NOT_ENGLISH:", "FORMAT:", "IN_FORMAT:"):
        assert key in prompt
    assert "OOD" not in prompt and "training" not in prompt.lower()
