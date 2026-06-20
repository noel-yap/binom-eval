"""End-to-end Claude evals for the example-skill.

For each entry in `evals.json` the `eval_runs` fixture runs `claude -p`
in adaptive concurrent batches and returns the parsed runs keyed by eval id.
This module grades them three ways:

  * `test_eval_assertion` -- one parametrized node per (eval, assertion),
    graded by the Beta-binomial posterior over the assertion's pass rate.
  * `test_eval_expectation` -- a per-eval rollup that, when any assertion
    fell short, fails once with the eval's `expected_output` for context.
  * `test_should_trigger_evals_invoked_skill` -- grades whether the
    `should_trigger` evals actually invoked the skill.

These carry the `live_eval` marker because each model call costs time and
money; select them with `-m live_eval`.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from binom_eval import (
    EvalRun,
    assert_eval_passed,
    eval_passed,
    failing_assertions,
    trial_outcomes,
    trigger_pass_counts,
)

from ._assertions import ASSERTION_HANDLERS

EVALS_PATH = Path(__file__).resolve().parent / "evals.json"


def _evals() -> list[dict]:
    """Load all eval entries from evals.json."""
    return json.loads(EVALS_PATH.read_text(encoding="utf-8"))["evals"]


def _assertion_params() -> list[pytest.param]:
    """Build a pytest.param per (eval_id, assertion_id) in evals.json."""
    return [
        pytest.param(ev["id"], ass["id"], id=f"{ev['id']}::{ass['id']}")
        for ev in _evals()
        for ass in ev["assertions"]
    ]


@pytest.mark.live_eval
@pytest.mark.parametrize("eval_id,assertion_id", _assertion_params())
def test_eval_assertion(
    eval_runs: dict[str, list[EvalRun]],
    live_eval_target_rate: float,
    eval_id: str,
    assertion_id: str,
) -> None:
    handler = ASSERTION_HANDLERS[assertion_id]
    outcomes = trial_outcomes(eval_runs[eval_id], handler)
    assert_eval_passed(
        outcomes, live_eval_target_rate, f"{eval_id}::{assertion_id}"
    )


@pytest.mark.live_eval
@pytest.mark.parametrize("eval_id", [ev["id"] for ev in _evals()])
def test_eval_expectation(
    eval_runs: dict[str, list[EvalRun]],
    live_eval_target_rate: float,
    eval_id: str,
) -> None:
    """Per-eval rollup: when any assertion failed the posterior bar, fail
    once with the eval's `expected_output` as the human-level intent."""
    ev = next(e for e in _evals() if e["id"] == eval_id)
    failing = failing_assertions(
        eval_runs[eval_id],
        ev["assertions"],
        ASSERTION_HANDLERS,
        live_eval_target_rate,
    )
    assert not failing, (
        f"{eval_id}: {len(failing)} assertion(s) below the bar "
        f"(P(rate >= {live_eval_target_rate:.3f}) must be >= 0.5):\n"
        + "\n".join(
            f"  - {aid}: {n}/{total} passed, p_good={p:.3f}"
            for aid, n, total, p in failing
        )
        + f"\n\nExpected outcome:\n  {ev['expected_output']}"
    )


@pytest.mark.live_eval
def test_should_trigger_evals_invoked_skill(
    eval_runs: dict[str, list[EvalRun]],
    live_eval_target_rate: float,
) -> None:
    counts = trigger_pass_counts(eval_runs, _evals())
    failures = [
        (eid, n, total)
        for eid, n, total in counts
        if not eval_passed(n, total, live_eval_target_rate)
    ]
    assert not failures, (
        f"example-skill invoked below the bar "
        f"(P(rate >= {live_eval_target_rate:.3f}) must be >= 0.5): "
        + ", ".join(f"{eid}: {n}/{total}" for eid, n, total in failures)
    )
