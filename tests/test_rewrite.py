from src.data.rewrite import judge_prompt, parse_judge, rewrite_prompt, select_examples


def _rows(texts: list[str]) -> list[dict]:
    return [{"id": f"r{i}", "text": t} for i, t in enumerate(texts)]


def test_select_examples_filters_length_and_language_and_is_seeded():
    ok = [" ".join(["word"] * n) for n in (10, 50, 120)]
    rows = _rows(ok + ["too short", " ".join(["word"] * 121), "Привет " * 20])
    picked = select_examples(rows, n=3, seed=0)
    assert sorted(picked) == sorted(ok)
    assert select_examples(rows, n=2, seed=0) == select_examples(rows, n=2, seed=0)


def test_rewrite_prompt_contains_examples_and_message():
    prompt = rewrite_prompt(["Example one here.", "Example two here."], "My landlord kept my deposit.")
    assert "Example one here." in prompt and "Example two here." in prompt
    assert prompt.rstrip().endswith("</message>")
    assert "My landlord kept my deposit." in prompt


def test_judge_prompt_shows_both_texts():
    prompt = judge_prompt("original text", "rewritten text")
    assert "original text" in prompt and "rewritten text" in prompt


def test_parse_judge():
    assert parse_judge("YES") is True
    assert parse_judge("  no, it drops the dates") is False
    assert parse_judge("**Yes**") is True
    assert parse_judge("I think so") is None
    assert parse_judge(None) is None
