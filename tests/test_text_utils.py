"""Unit tests for `binom_eval.text_utils` (assertion text helpers).

`code_blocks` and `first_line` are exercised through the per-skill
`_assertions.py` suites; the skill-independent `missing_from` is tested here.
"""

from __future__ import annotations

from binom_eval import missing_from


class TestMissingFrom:
    def test_returns_needles_absent_from_haystack(self) -> None:
        assert missing_from(("a", "z"), "abc") == ["z"]

    def test_returns_empty_when_all_present(self) -> None:
        assert missing_from(("a", "b"), "abc") == []

    def test_returns_all_when_none_present(self) -> None:
        assert missing_from(("x", "y"), "abc") == ["x", "y"]

    def test_preserves_original_order(self) -> None:
        assert missing_from(("z", "y", "x"), "") == ["z", "y", "x"]

    def test_empty_needles_returns_empty(self) -> None:
        assert missing_from((), "abc") == []