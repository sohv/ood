# unit tests for iterative_discovery discovery helpers: naming parse, prompt construction, gap matching.
# uv run --no-sync -m pytest tests/test_discovery.py -v

import json

import numpy as np

from src.metrics.discovery import (
    NAMING_PROMPT_SHARE,
    centroid,
    match_gap,
    name_clusters,
    naming_prompt,
    parse_label,
    parse_naming,
)


def test_parse_naming_reads_all_fields():
    out = parse_naming("NAME: Fan fiction crossovers\nDESCRIPTION: Stories mixing franchises.\nCOHERENT: YES")
    assert out == {
        "name": "Fan fiction crossovers",
        "description": "Stories mixing franchises.",
        "coherent": True,
        "shared": None,
    }


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


class Reply:
    def __init__(self, content: str | None):
        self.choices = [type("Choice", (), {"message": type("Message", (), {"content": content})()})()]


class ScriptedCompletions:
    def __init__(self, contents: list[str | None]):
        self.contents, self.calls = list(contents), 0

    async def create(self, **kwargs):
        self.calls += 1
        return Reply(self.contents.pop(0))


class ScriptedClient:
    """Returns scripted replies in order. Exercises the empty-reply boundary, not the model."""

    def __init__(self, contents: list[str | None]):
        self.chat = type("Chat", (), {"completions": ScriptedCompletions(contents)})()


async def test_name_clusters_never_caches_an_empty_reply(tmp_path):
    client = ScriptedClient([None, "", None, "NAME: x\nDESCRIPTION: y\nCOHERENT: YES"])
    out = await name_clusters(client, "m", ["p", "q"], str(tmp_path), max_concurrent=1)
    # p gets three empty replies and stays null; q gets the real one
    assert out == [None, "NAME: x\nDESCRIPTION: y\nCOHERENT: YES"]
    assert client.chat.completions.calls == 4
    assert len(list(tmp_path.glob("*.json"))) == 1


async def test_name_clusters_retries_an_empty_reply(tmp_path):
    client = ScriptedClient([None, "NAME: x\nDESCRIPTION: y\nCOHERENT: NO"])
    out = await name_clusters(client, "m", ["p"], str(tmp_path))
    assert out == ["NAME: x\nDESCRIPTION: y\nCOHERENT: NO"]
    assert json.loads(next(tmp_path.glob("*.json")).read_text())["response"] == out[0]


def test_parse_label_takes_the_first_allowed_label():
    labels = ["LEGAL", "MEDICAL", "CODE", "OTHER"]
    assert parse_label("CODE", labels) == "CODE"
    assert parse_label("Answer: legal.", labels) == "LEGAL"
    assert parse_label("MEDICAL, not LEGAL", labels) == "MEDICAL"
    assert parse_label("codex", labels) is None
    assert parse_label(None, labels) is None


def test_parse_naming_reads_shared_count():
    out = parse_naming("NAME: Legal questions\nDESCRIPTION: Tenancy and work.\nSHARED: 17")
    assert out["shared"] == 17 and out["coherent"] is None
    assert parse_naming("NAME: x\nDESCRIPTION: y\nSHARED: about a dozen")["shared"] is None


def test_naming_prompt_share_template_asks_for_a_count():
    prompt = naming_prompt(["a", "b", "c"], n=25, seed=0, template=NAMING_PROMPT_SHARE)
    assert "SHARED:" in prompt and "of the 3 messages" in prompt
