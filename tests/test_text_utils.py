"""Unit tests for `binom_eval.text_utils` (assertion text helpers).

`fenced_blocks` and `code_blocks` are covered here, including the
citation-fence desynchronization regression; `first_line` is exercised
through the per-skill `_assertions.py` suites, and the skill-independent
`missing_from`, `comment_mark_re`, `marked_regions`, and
`comment_sections` are tested here.
"""

from __future__ import annotations

from binom_eval import (
    code_blocks,
    comment_mark_re,
    comment_sections,
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

    def test_does_not_match_indented_comment(self) -> None:
        pattern = comment_mark_re("pure core")
        assert pattern.search("  // Pure core: compute") is None

    def test_does_not_match_trailing_same_line_comment(self) -> None:
        pattern = comment_mark_re("pure core")
        assert pattern.search("const x = 1; // pure core") is None

    def test_does_not_match_phrase_only_in_description(self) -> None:
        pattern = comment_mark_re("pure core")
        text = "// Imperative shell: calls the pure core and applies effects"
        assert pattern.search(text) is None

    def test_matches_decorated_opener_with_no_colon(self) -> None:
        pattern = comment_mark_re("pure core")
        assert pattern.search("// --- pure core ---") is not None

    def test_matches_bare_opener_with_no_colon(self) -> None:
        pattern = comment_mark_re("pure core")
        assert pattern.search("// PURE CORE") is not None


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

    def test_indented_narration_comment_does_not_open_region(self) -> None:
        text = (
            "function decide(x) {\n"
            "  // Pure core: compute the decision\n"
            "  const d = f(x);\n"
            "  await db.write();\n"
            "}\n"
        )
        assert marked_regions(text, "pure core") == []

    def test_phrase_only_in_description_does_not_open_region(self) -> None:
        text = (
            "// pure core: data in, data out\n"
            "code\n"
            "// end pure core\n"
            "// Imperative shell: calls the pure core and applies effects\n"
            "await db.write();\n"
        )
        regions = marked_regions(text, "pure core")
        assert len(regions) == 1
        assert "code" in regions[0]
        assert "await" not in regions[0]


class TestCommentSections:
    def test_two_headed_sections_in_order(self) -> None:
        text = (
            "// Header A\n"
            "body a1\n"
            "body a2\n"
            "// Header B\n"
            "body b1\n"
        )
        assert comment_sections(text) == [
            ("// Header A", "body a1\nbody a2"),
            ("// Header B", "body b1"),
        ]

    def test_consecutive_comment_lines_form_one_header(self) -> None:
        text = "// Header line 1\n// Header line 2\nbody\n"
        assert comment_sections(text) == [
            ("// Header line 1\n// Header line 2", "body")
        ]

    def test_indented_comment_stays_in_body_and_does_not_split(self) -> None:
        text = "// Header\ncode1\n  // note\ncode2\n"
        assert comment_sections(text) == [
            ("// Header", "code1\n  // note\ncode2")
        ]

    def test_text_before_first_header_is_excluded(self) -> None:
        text = "preamble\nmore preamble\n// Header\nbody\n"
        assert comment_sections(text) == [("// Header", "body")]

    def test_pure_section_stops_at_shell_header(self) -> None:
        text = (
            "// Pure function: compute the decision\n"
            "const d = f(x);\n"
            "// Imperative shell: perform I/O\n"
            "await db.write(d);\n"
        )
        sections = comment_sections(text)
        assert len(sections) == 2
        pure_header, pure_body = sections[0]
        shell_header, shell_body = sections[1]
        assert "pure" in pure_header.lower()
        assert "shell" in shell_header.lower()
        assert "const d = f(x);" in pure_body
        assert "await" not in pure_body
        assert "await db.write(d);" in shell_body


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
