"""Unit tests for `binom_eval.text_utils` (assertion text helpers).

`code_blocks` and `first_line` are exercised through the per-skill
`_assertions.py` suites; the skill-independent `missing_from`,
`comment_mark_re`, and `marked_regions` are tested here.
"""

from __future__ import annotations

from binom_eval import (
    comment_mark_re,
    contains,
    contains_all,
    has_code_blocks,
    marked_regions,
    missing_from,
)


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


class TestContains:
    def test_true_when_needle_present(self) -> None:
        assert contains("abc", "b") is True

    def test_false_when_needle_absent(self) -> None:
        assert contains("abc", "z") is False


class TestContainsAll:
    def test_true_when_all_present(self) -> None:
        assert contains_all("abc", ("a", "b")) is True

    def test_false_when_any_missing(self) -> None:
        assert contains_all("abc", ("a", "z")) is False


class TestHasCodeBlocks:
    def test_true_when_fenced_block_present(self) -> None:
        assert has_code_blocks("```ts\nconst x = 1\n```") is True

    def test_false_when_no_fence(self) -> None:
        assert has_code_blocks("plain text") is False


class TestCommentMarkRe:
    def test_matches_plain_opener(self) -> None:
        pattern = comment_mark_re("pure core")
        assert pattern.search("// pure core: data in, data out") is not None

    def test_matches_decorated_opener(self) -> None:
        pattern = comment_mark_re("pure core")
        assert (
            pattern.search("// --- pure core: data in, data out ---")
            is not None
        )

    def test_does_not_match_lone_end_marker(self) -> None:
        pattern = comment_mark_re("pure core")
        assert pattern.search("// --- end pure core ---") is None

    def test_does_not_match_text_with_no_marker(self) -> None:
        pattern = comment_mark_re("pure core")
        assert pattern.search("export function f() {}") is None


class TestMarkedRegions:
    def test_plain_marker_excludes_trailing_shell(self) -> None:
        text = "// pure core: data in, data out\ncode\n// end pure core\nshell"
        regions = marked_regions(text, "pure core")
        assert len(regions) == 1
        assert "code" in regions[0]
        assert "shell" not in regions[0]

    def test_decorated_marker_excludes_trailing_shell(self) -> None:
        text = (
            "// --- pure core: data in, data out, no I/O ---\n"
            "code\n"
            "// --- end pure core ---\n"
            "export async function processOrder() {"
            " await db.getOrder(); }"
        )
        regions = marked_regions(text, "pure core")
        assert len(regions) == 1
        assert "code" in regions[0]
        assert "await" not in regions[0]

    def test_unclosed_marker_runs_to_end_of_text(self) -> None:
        text = "// Pure core\ncode"
        regions = marked_regions(text, "pure core")
        assert regions == ["code"]

    def test_lone_end_marker_is_not_an_opener(self) -> None:
        text = "// --- end pure core ---\nshell"
        regions = marked_regions(text, "pure core")
        assert regions == []

    def test_multiple_regions_are_returned_in_order(self) -> None:
        text = (
            "// pure core: a\n"
            "first\n"
            "// end pure core\n"
            "shell\n"
            "// pure core: b\n"
            "second\n"
            "// end pure core\n"
        )
        regions = marked_regions(text, "pure core")
        assert len(regions) == 2
        assert "first" in regions[0]
        assert "second" in regions[1]
