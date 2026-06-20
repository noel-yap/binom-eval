"""Text and regex utilities shared by per-skill `_assertions.py` modules.

These are pure string helpers with no dependency on the eval-running
machinery: extracting fenced code blocks from model output, taking a
capped first line for messages, and substring-presence checks. The
function-definition regexes (`NAMED_FN_RE`, `ARROW_FN_RE`) are exposed so
per-skill assertions can find declared functions in refactored TypeScript.
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