"""Unit tests for `binom_eval.text_utils` (assertion text helpers).

`fenced_blocks` and `code_blocks` are covered here, including the
citation-fence desynchronization regression; `first_line` is exercised
through the per-skill `_assertions.py` suites, and the skill-independent
`missing_from`, `comment_mark_re`, and `marked_regions` are tested here.
"""

from __future__ import annotations

from binom_eval import (
    code_blocks,
    comment_mark_re,
    contains,
    contains_all,
    fenced_blocks,
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


_CITATION_THEN_TS = """\
Cited context:

```12:15:src/app/checkout.ts
const service = new InvoiceService();
```

Final contents:

```typescript
const service = new InvoiceService(new TaxRateClient());
```
"""


class TestFencedBlocks:
    def test_returns_info_and_body_pairs(self) -> None:
        assert fenced_blocks("```ts\nconst x = 1\n```") == [
            ("ts", "const x = 1")
        ]

    def test_bare_fence_has_empty_info(self) -> None:
        assert fenced_blocks("```\nplain\n```") == [("", "plain")]

    def test_citation_fence_does_not_desynchronize(self) -> None:
        assert fenced_blocks(_CITATION_THEN_TS) == [
            (
                "12:15:src/app/checkout.ts",
                "const service = new InvoiceService();",
            ),
            (
                "typescript",
                "const service = new InvoiceService(new TaxRateClient());",
            ),
        ]

    def test_prose_between_blocks_is_not_captured(self) -> None:
        blocks = fenced_blocks(_CITATION_THEN_TS)
        assert "Final contents" not in blocks[0][1]
        assert "Final contents" not in blocks[1][1]

    def test_unclosed_block_is_dropped(self) -> None:
        assert fenced_blocks("```ts\nconst x = 1") == []

    def test_multi_word_info_fence_stays_in_sync(self) -> None:
        text = "```ts twoslash\ncode\n```\n```typescript\nreal\n```"
        assert fenced_blocks(text) == [
            ("ts twoslash", "code"),
            ("typescript", "real"),
        ]

    def test_four_backtick_line_is_not_a_fence(self) -> None:
        assert fenced_blocks("````\nx\n````") == []


class TestCodeBlocks:
    def test_extracts_ts_typescript_and_bare_blocks(self) -> None:
        text = "```ts\na\n```\n```typescript\nb\n```\n```\nc\n```"
        assert code_blocks(text) == ["a", "b", "c"]

    def test_skips_citation_fences_without_desync(self) -> None:
        assert code_blocks(_CITATION_THEN_TS) == [
            "const service = new InvoiceService(new TaxRateClient());"
        ]

    def test_skips_json_blocks(self) -> None:
        assert code_blocks('```json\n{"a": 1}\n```') == []

    def test_extracts_block_with_multi_word_ts_info(self) -> None:
        assert code_blocks("```ts twoslash\ncode\n```") == ["code"]

    def test_skips_block_with_multi_word_non_ts_info(self) -> None:
        assert code_blocks("```json schema\n{}\n```") == []


class TestHasCodeBlocksCountsAnyFence:
    def test_true_for_citation_fenced_block(self) -> None:
        assert has_code_blocks("```1:4:src/x.ts\ncode\n```") is True
