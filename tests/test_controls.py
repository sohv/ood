import pytest

from src.data.controls import dedupe, take_non_members
from src.data.pools import MembershipIndex


def test_dedupe_normalises_case_and_whitespace_and_drops_empty():
    assert dedupe(["Hello  World", "hello world", "", "  ", None, "other"]) == ["Hello  World", "other"]


def test_take_non_members_skips_members_and_counts_them():
    index = MembershipIndex(["the mix row one", "the mix row two"])
    texts = ["the mix row one", "fresh text a", "the mix row two", "fresh text b", "fresh text c"]
    kept, checked, removed = take_non_members(texts, index, n=3, seed=0)
    assert sorted(kept) == ["fresh text a", "fresh text b", "fresh text c"]
    assert removed == checked - 3


def test_take_non_members_raises_when_short():
    index = MembershipIndex(["only row"])
    with pytest.raises(ValueError):
        take_non_members(["only row", "new"], index, n=2, seed=0)
