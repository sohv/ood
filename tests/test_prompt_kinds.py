from src.generation.prompt_kinds import (
    LLM_KINDS,
    PROMPT,
    VALUES,
    has_placeholder,
    rule_tags,
    show,
    swap_placeholders,
    template_flags,
)
from src.generation.request_grader import parse_grade


def test_placeholder_rule():
    assert has_placeholder("Write a letter to NAME_1 about NAME_2.")
    assert has_placeholder("Dear [NAME], thanks")
    assert not has_placeholder("My name is Anna and NAMES are hard")


def test_template_needs_shared_full_prefix():
    shared = "Given the document below, you have to determine if the summary"
    texts = [f"{shared} {i}" for i in range(5)] + ["short", "short", "short"]
    flags = template_flags(texts, prefix_chars=40, min_count=5)
    assert flags[:5] == [True] * 5
    assert flags[5:] == [False] * 3


def test_rule_tags_long():
    tags = rule_tags(["a" * 2000, "b"])
    assert tags[0]["long"] and not tags[1]["long"]


def test_prompt_formats_and_parses():
    prompt = PROMPT.format(user=show("x" * 3000, 1500))
    assert "first 1,500 shown" in prompt and "3,000 characters" in prompt
    reply = "{" + ", ".join(f'"{k}": "no"' for k in LLM_KINDS) + "}"
    assert parse_grade(reply, VALUES) == {k: "no" for k in LLM_KINDS}


def test_swap_placeholders_is_consistent():
    assert swap_placeholders("NAME_1 met NAME_2, then NAME_1 left.") == "James met Mary, then James left."
    assert swap_placeholders("no names here") == "no names here"
