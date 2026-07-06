"""Text and regex utilities shared by per-skill `_assertions.py` modules.

These are pure string helpers with no dependency on the eval-running
machinery: extracting fenced code blocks from model output, taking a
capped first line for messages, and substring-presence checks. The
function-definition regexes (`NAMED_FN_RE`, `ARROW_FN_RE`) are exposed so
per-skill assertions can find declared functions in refactored TypeScript.
`comment_mark_re` and `marked_regions` locate decoration-tolerant
`// ... <phrase> ... // ... end <phrase>` comment markers, for skills that
ask a model to delimit a region (e.g. a "pure core") with comments.
"""

from __future__ import annotations

import re

CODE_BLOCK_RE = re.compile(r"```(?:ts|typescript)?\n(.*?)```", re.DOTALL)
NAMED_FN_RE = re.compile(r"\bfunction\s+(\w+)\s*\(")
ARROW_FN_RE = re.compile(
    r"\bconst\s+(\w+)\s*(?::[^=]+)?=\s*(?:\([^)]*\)|\w+)\s*(?::[^=]+)?=>"
)


def code_blocks(text: str) -> list[str]:
    """Extract bodies of fenced ```ts / ```typescript / ``` code blocks."""
    return CODE_BLOCK_RE.findall(text)


def first_line(block: str) -> str:
    """First non-empty line of `block`, capped at 80 chars for messages."""
    return next(iter(block.strip().splitlines()), "")[:80]


def missing_from(needles: tuple[str, ...], haystack: str) -> list[str]:
    """Return the substrings (needles) not found within a larger string.

    Each needle is checked with a plain substring test, so it matches
    anywhere in the haystack with no word-boundary requirement. The
    returned list preserves the original needle order, not the order the
    needles appear in the haystack.

    Args:
      needles: The substrings to look for.
      haystack: The string to search within.

    Returns:
      The needles absent from haystack, in their original order. An empty
      list means every needle was present.

    Example:
      >>> missing_from(("a", "z"), "abc")
      ['z']
      >>> missing_from(("a", "b"), "abc")
      []
    """
    return list(filter(lambda n: n not in haystack, needles))


def contains(haystack: str, needle: str) -> bool:
    """True when ``needle`` appears in ``haystack``."""
    return needle in haystack


def contains_all(haystack: str, needles: tuple[str, ...]) -> bool:
    """True when every ``needles`` entry appears in ``haystack``."""
    return not missing_from(needles, haystack)


def has_code_blocks(text: str) -> bool:
    """True when ``text`` contains at least one fenced code block."""
    return bool(code_blocks(text))


def comment_mark_re(phrase: str) -> re.Pattern[str]:
    """Compile a decoration-tolerant regex for a `// ... <phrase> ...` comment.

    Matches a line comment containing `phrase` (case-insensitive, words
    separated by whitespace, word-bounded) regardless of decoration such
    as `// --- pure core ---`. Lines marking the END of a region
    (`// ... end <phrase>`) are excluded, so the opener regex never
    matches a closing marker.

    Args:
      phrase: The words to look for, e.g. "pure core".

    Returns:
      A compiled, case-insensitive regex matching an opening marker line.
    """
    words = r"\s+".join(map(re.escape, phrase.split()))
    return re.compile(
        rf"//(?![^\n]*\bend\s+{words}\b)[^\n]*?\b{words}\b",
        re.IGNORECASE,
    )


def marked_regions(text: str, phrase: str) -> list[str]:
    """Regions between `// ... <phrase>` and `// ... end <phrase>` comments.

    Each region runs from the line after a decoration-tolerant opening
    marker (see `comment_mark_re`) to the matching `// ... end <phrase>`
    closing marker, or to the end of `text` when the close is omitted.

    Args:
      text: The model output to search.
      phrase: The words identifying the marked region, e.g. "pure core".

    Returns:
      The text of each marked region, in the order they appear in `text`.
    """
    words = r"\s+".join(map(re.escape, phrase.split()))
    region_re = re.compile(
        rf"//(?![^\n]*\bend\s+{words}\b)[^\n]*?\b{words}\b[^\n]*\n"
        rf"(.*?)(?=//[^\n]*?\bend\s+{words}\b|\Z)",
        re.IGNORECASE | re.DOTALL,
    )
    return region_re.findall(text)
