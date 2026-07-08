"""Text and regex utilities shared by per-skill `_assertions.py` modules.

These are pure string helpers with no dependency on the eval-running
machinery: extracting fenced code blocks from model output, taking a
capped first line for messages, and substring-presence checks. The
function-definition regexes (`NAMED_FN_RE`, `ARROW_FN_RE`) are exposed so
per-skill assertions can find declared functions in refactored TypeScript.
`comment_mark_re` and `marked_regions` locate decoration-tolerant
`// ... <phrase> ... // ... end <phrase>` comment markers, for skills that
ask a model to delimit a region (e.g. a "pure core") with comments.
`comment_sections` is the more general primitive: it splits a block into
`(header, body)` pairs at each unindented `//` comment line, for skills
whose sections are labeled with arbitrary wording rather than a single
known phrase.
"""

from __future__ import annotations

import re

NAMED_FN_RE = re.compile(r"\bfunction\s+(\w+)\s*\(")
ARROW_FN_RE = re.compile(
    r"\bconst\s+(\w+)\s*(?::[^=]+)?=\s*(?:\([^)]*\)|\w+)\s*(?::[^=]+)?=>"
)

# A fence line: up to three spaces of indent, ``` and the rest of the line
# as the info string (CommonMark reads the whole remainder, so multi-word
# tags like `ts twoslash` and arbitrary tags such as Cursor's
# `start:end:path.ts` citation fences all open a block). `[^`]*` keeps
# lines of four or more backticks from matching. A fence line whose info
# string strips to empty closes the open block; a closing fence never
# carries an info string.
_FENCE_RE = re.compile(r"^ {0,3}```([^`]*)$")

# Info strings `code_blocks` treats as TypeScript output.
_TS_INFOS = frozenset({"", "ts", "typescript"})


def fenced_blocks(text: str) -> list[tuple[str, str]]:
    """Every fenced code block in `text`, as `(info, body)` pairs.

    Fences are parsed line-wise, the way CommonMark treats them: any
    ``` line opens a block whatever its info string, and the next bare
    ``` line closes it (a fence-like line carrying an info string inside
    an open block is body content, since per CommonMark a closing fence
    has no info string). Parsing line-wise is what keeps unknown tags
    from desynchronizing extraction -- a regex that recognizes only known
    language tags as openers skips e.g. a `start:end:path.ts` citation
    fence (as emitted by Cursor), mistakes that block's closing fence for
    an opener, and then captures the prose between real blocks instead of
    the code. An unterminated final block is dropped.

    Args:
      text: Model output that may contain fenced code blocks.

    Returns:
      One `(info, body)` pair per closed block, in order of appearance.
      `info` is the fence's full info string, stripped of surrounding
      whitespace but otherwise verbatim -- multi-word info strings such
      as `ts twoslash` are allowed, and a bare fence yields `""`; `body`
      carries no trailing newline.
    """
    blocks: list[tuple[str, str]] = []
    info: str | None = None
    body: list[str] = []
    for line in text.splitlines():
        fence = _FENCE_RE.match(line)
        if info is None:
            if fence:
                info = fence.group(1).strip()
                body = []
        elif fence and not fence.group(1).strip():
            blocks.append((info, "\n".join(body)))
            info = None
        else:
            body.append(line)
    return blocks


def code_blocks(text: str) -> list[str]:
    """Extract bodies of fenced ```ts / ```typescript / ``` code blocks.

    Classification looks at the info string's first word, so a
    multi-word tag such as ```ts twoslash still counts as TypeScript.
    Blocks whose first word is anything else (```json, a citation
    fence, ...) are skipped -- but skipped *correctly*, so the
    TypeScript blocks around them are still extracted (see
    `fenced_blocks`).
    """
    return [
        body
        for fence_info, body in fenced_blocks(text)
        if (fence_info.split() or [""])[0].lower() in _TS_INFOS
    ]


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
    """True when ``text`` contains at least one closed fenced code block.

    Counts a block with any info string: a response whose only code sits
    inside a ```json or citation fence still has a code block. Whether
    the right kind of block is present is the caller's stricter check,
    via `code_blocks`.
    """
    return bool(fenced_blocks(text))


def comment_mark_re(phrase: str) -> re.Pattern[str]:
    """Compile a decoration-tolerant regex for a `// <phrase> ...` comment.

    Matches a line comment whose LABEL -- the text before the comment's
    first `:` -- contains `phrase` (case-insensitive, words separated by
    whitespace, word-bounded), regardless of decoration such as
    `// --- pure core ---`. A marker must NAME its region: a comment that
    merely mentions `phrase` in its description after the colon (e.g.
    `// Imperative shell: calls the pure core and applies effects`) is not
    a marker and never matches. Lines marking the END of a region
    (`// ... end <phrase>`) are excluded, so the opener regex never
    matches a closing marker.

    A section marker is an UNINDENTED comment line -- the `//` must begin
    the line. An indented narration comment (`  // pure core: ...`) inside
    a function body, or a trailing same-line comment
    (`const x = 1; // pure core`), is not a marker and never matches.

    Args:
      phrase: The words to look for, e.g. "pure core".

    Returns:
      A compiled, case-insensitive regex matching an opening marker line.
    """
    words = r"\s+".join(map(re.escape, phrase.split()))
    return re.compile(
        rf"^//(?![^\n]*\bend\s+{words}\b)[^:\n]*\b{words}\b",
        re.IGNORECASE | re.MULTILINE,
    )


def marked_regions(text: str, phrase: str) -> list[str]:
    """Regions between `// <phrase>` and `// ... end <phrase>` comments.

    Each region runs from the line after a decoration-tolerant opening
    marker (see `comment_mark_re`) to the matching `// ... end <phrase>`
    closing marker, or to the end of `text` when the close is omitted.

    A marker must NAME its region: `phrase` has to appear in the
    comment's LABEL, the text before its first `:`. A comment that merely
    mentions `phrase` in its description after the colon (e.g.
    `// Imperative shell: calls the pure core and applies effects`) does
    not open a region.

    A section marker is an UNINDENTED comment line -- the `//` must begin
    the line. An indented narration comment (`  // pure core: ...`) inside
    a function body, or a trailing same-line comment
    (`const x = 1; // pure core`), is not a marker and never opens or
    closes a region.

    Args:
      text: The model output to search.
      phrase: The words identifying the marked region, e.g. "pure core".

    Returns:
      The text of each marked region, in the order they appear in `text`.
    """
    words = r"\s+".join(map(re.escape, phrase.split()))
    region_re = re.compile(
        rf"^//(?![^\n]*\bend\s+{words}\b)[^:\n]*\b{words}\b[^\n]*\n"
        rf"(.*?)(?=^//[^\n]*?\bend\s+{words}\b|\Z)",
        re.IGNORECASE | re.DOTALL | re.MULTILINE,
    )
    return region_re.findall(text)


def comment_sections(text: str) -> list[tuple[str, str]]:
    """Split `text` into sections headed by unindented `//` comment lines.

    A header is one or more consecutive lines each starting with `//` at
    column 0 (joined with newlines); its section body runs until the next
    header or the end of `text`. Text before the first header is not
    returned (it has no header to classify it by).

    Indented `//` comments (anything other than a column-0 `//`) belong
    to the current section body -- they never start or split a section.

    Args:
      text: The model output to split into headed sections.

    Returns:
      (header, body) pairs in document order.
    """
    sections: list[tuple[str, str]] = []
    header: list[str] = []
    body: list[str] = []
    in_body = False
    for line in text.splitlines():
        if line.startswith("//"):
            if in_body:
                sections.append(("\n".join(header), "\n".join(body)))
                header, body = [], []
                in_body = False
            header.append(line)
        elif header:
            body.append(line)
            in_body = True
    if header:
        sections.append(("\n".join(header), "\n".join(body)))
    return sections
