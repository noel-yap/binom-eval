"""Assertion handlers for the example-skill eval suite.

Each handler takes an `EvalRun` and raises `AssertionFailure` when the run
fails that assertion; a clean return is a pass. The ids must match the
`assertions[].id` values in `evals.json` -- `make_eval_runs_fixture` runs
`assert_handler_coverage` at load time, so a missing handler fails fast.

These illustrate the two common shapes: structural checks over the proposed
output (`code_blocks`) and substring-presence checks (`missing_from`).
Add labeled ``sections`` when a one-line summary is not enough to debug
the failure.
"""

from __future__ import annotations

from binom_eval import AssertionFailure, EvalRun, code_blocks, missing_from

# A real skill would assert on its own domain-specific marker.
EXPECTED_MARKER = "example-marker"


def _emits_code_block(run: EvalRun) -> None:
    """Assert the assistant produced at least one fenced code block."""
    if not code_blocks(run.assistant_text):
        raise AssertionFailure(
            "no fenced code block in assistant output",
            sections=(
                ("Assistant reply", run.assistant_text or "(empty)"),
            ),
        )


def _code_block_has_marker(run: EvalRun) -> None:
    """Assert some fenced code block contains the expected marker token."""
    blocks = code_blocks(run.assistant_text)
    if all(missing_from((EXPECTED_MARKER,), block) for block in blocks):
        raise AssertionFailure(
            f"no code block contained the marker {EXPECTED_MARKER!r}",
            sections=(
                ("Expected marker", EXPECTED_MARKER),
                (
                    "Code blocks",
                    "\n\n---\n\n".join(blocks) if blocks else "(none)",
                ),
            ),
        )


ASSERTION_HANDLERS = {
    "emits-code-block": _emits_code_block,
    "code-block-has-marker": _code_block_has_marker,
}
