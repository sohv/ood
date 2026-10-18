# unit tests for exp09 discovery helpers: naming parse, prompt construction, gap matching.
# uv run --no-sync -m pytest tests/test_discovery.py -v

import numpy as np

from src.metrics.discovery import centroid, match_gap, naming_prompt, parse_naming


def test_parse_naming_reads_all_fields():
    out = parse_naming("NAME: Fan fiction crossovers\nDESCRIPTION: Stories mixing franchises.\nCOHERENT: YES")
    assert out == {"name": "Fan fiction crossovers", "description": "Stories mixing franchises.", "coherent": True}


def test_parse_naming_handles_missing_reply_and_no():
    assert parse_naming(None)["coherent"] is None
    assert parse_naming("NAME: Mixed\nDESCRIPTION: Nothing shared.\nCOHERENT: NO")["coherent"] is False


def test_naming_prompt_truncates_and_never_mentions_ood():
    prompt = naming_prompt(["word " * 200, "short"], n=25, seed=0)
    assert "OOD" not in prompt and "training" not in prompt.lower()
    assert max(len(chunk) for chunk in prompt.split("\n---\n")[1:-1] or [""]) <= 400


def test_match_gap_uses_threshold():
    a = centroid(np.array([[1.0, 0.0], [0.9, 0.1]]))
    b = centroid(np.array([[0.0, 1.0]]))
    assert match_gap(a, [b, a], 0.85) == 1
    assert match_gap(b, [a], 0.85) is None
