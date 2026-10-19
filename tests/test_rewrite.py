from src.data.rewrite import (
    ban_list,
    find_leaks,
    judge_prompt,
    lexical_rewrite_prompt,
    parse_judge,
    parse_topic,
    repair_prompt,
    rewrite_prompt,
    select_examples,
    topic_prompt,
)


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


def test_ban_list_takes_both_directions_and_drops_stop_words():
    pos = ["landlord deposit lease the tenant"] * 20 + ["landlord eviction court"] * 20
    neg = ["write story python code the"] * 20 + ["write poem essay"] * 20
    banned = ban_list(pos, neg, n_top=5, seed=0)
    assert {"landlord", "write"} <= set(banned)
    assert "the" not in banned
    assert banned == sorted(set(banned))
    assert all(t.isalpha() and len(t) >= 3 for t in banned)


def test_lexical_rewrite_prompt_lists_banned_words():
    prompt = lexical_rewrite_prompt(["Example one here."], "My landlord kept my deposit.", ["deposit", "landlord"])
    assert "deposit, landlord" in prompt
    assert "Example one here." in prompt and "My landlord kept my deposit." in prompt


def test_find_leaks_matches_whole_lowercased_tokens():
    assert find_leaks("My LANDLORD kept it.", ["landlord", "lease"]) == ["landlord"]
    assert find_leaks("Landlords are people", ["landlord"]) == []
    assert find_leaks("nothing here", ["landlord"]) == []


def test_topic_prompt_and_parse_topic():
    assert "My knee hurts." in topic_prompt("My knee hurts.")
    assert parse_topic("LAWYER") == "lawyer"
    assert parse_topic("**Doctor** because") == "doctor"
    assert parse_topic("neither.") == "neither"
    assert parse_topic("I am not sure") is None
    assert parse_topic(None) is None


def test_repair_prompt_names_the_leaked_words_and_keeps_the_ban_list():
    prompt = repair_prompt(["Example one."], "My landlord kept it.", ["landlord", "lease"], ["landlord"])
    assert "My landlord kept it." in prompt and "landlord, lease" in prompt
    assert prompt.rstrip().endswith("still uses these words, which are not allowed: landlord")
