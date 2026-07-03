"""Consumer helpers for binom-eval eval suites.

Per-suite directories keep domain-specific ``evals.json`` and
``_assertions.py``; this module binds the shared pytest wiring so
``conftest.py`` and ``test_evals.py`` stay thin.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Any, Literal

import pytest

from binom_eval.grading import (
    eval_passed,
    expand_evals,
    failing_assertions,
    graded_runs,
    load_evals,
    trial_outcomes,
    trial_outcomes_failure_message,
    trial_outcomes_passed,
    trigger_pass_counts,
)
from binom_eval.plugin import make_eval_runs_fixture
from binom_eval.stream_json import EvalRun, agent_invoked

TriggerMode = Literal["skill", "agent"]


def bind_eval_runs_fixture(
    eval_dir: Path,
    subject_name: str,
    handlers: dict[str, Callable[[EvalRun], None]],
    *,
    repo_root: Path | None = None,
) -> Callable[..., dict[str, list[EvalRun]]]:
    """Return a session-scoped ``eval_runs`` fixture for an eval directory.

    ``eval_dir`` is the directory containing ``evals.json`` (typically the
    suite's ``evals/`` folder). When ``repo_root`` is omitted, ``claude -p``
    runs with ``eval_dir`` as the working tree (the bundled example pattern);
    pass an explicit repo root when prompts reference files elsewhere.
    """
    eval_dir = Path(eval_dir).resolve()
    evals_path = eval_dir / "evals.json"
    root = eval_dir if repo_root is None else Path(repo_root).resolve()
    return make_eval_runs_fixture(evals_path, root, subject_name, handlers)


def _assertion_params(evals: list[dict[str, Any]]) -> list[pytest.param]:
    return [
        pytest.param(ev["id"], ass["id"], id=f"{ev['id']}::{ass['id']}")
        for ev in evals
        for ass in ev.get("assertions", [])
    ]


def _agent_trigger_pass_counts(
    runs: dict[str, list[EvalRun]],
    evals_path: Path,
    agent_name: str,
    trigger_assertion: str,
) -> list[tuple[str, int, int]]:
    positive = [
        ev
        for ev in expand_evals(evals_path)
        if any(a["id"] == trigger_assertion for a in ev.get("assertions", []))
    ]
    return [
        (
            ev["id"],
            sum(
                agent_invoked(r, agent_name)
                for r in graded_runs(runs[ev["id"]])
            ),
            len(graded_runs(runs[ev["id"]])),
        )
        for ev in positive
    ]


def register_live_eval_tests(
    namespace: ModuleType | dict[str, Any],
    *,
    evals_path: Path,
    handlers: dict[str, Callable[[EvalRun], None]],
    subject_name: str,
    trigger: TriggerMode = "skill",
    agent_trigger_assertion: str = "invokes-agent",
) -> None:
    """Attach the standard live-eval test functions to a suite's test module.

    Pass ``globals()`` from the suite's ``test_evals.py``. Registers three
    pytest nodes:

      * ``test_eval_assertion`` -- one parametrized node per (eval, assertion)
      * ``test_eval_expectation`` -- per-eval rollup with ``expected_output``
      * ``test_should_trigger_evals_invoked_skill`` or
        ``test_should_invoke_agent_evals`` -- trigger rollup for skill or agent
    """
    evals_path = Path(evals_path).resolve()
    evals = load_evals(evals_path)
    if isinstance(namespace, dict):
        module_name = namespace["__name__"]
    else:
        module_name = namespace.__name__

    @pytest.mark.live_eval
    @pytest.mark.parametrize("eval_id,assertion_id", _assertion_params(evals))
    def test_eval_assertion(
        eval_runs: dict[str, list[EvalRun]],
        live_eval_target_rate: float,
        eval_id: str,
        assertion_id: str,
    ) -> None:
        handler = handlers[assertion_id]
        outcomes = trial_outcomes(eval_runs[eval_id], handler)
        label = f"{eval_id}::{assertion_id}"
        assert trial_outcomes_passed(outcomes, live_eval_target_rate), (
            trial_outcomes_failure_message(
                outcomes, live_eval_target_rate, label
            )
        )

    @pytest.mark.live_eval
    @pytest.mark.parametrize("eval_id", [ev["id"] for ev in evals])
    def test_eval_expectation(
        eval_runs: dict[str, list[EvalRun]],
        live_eval_target_rate: float,
        eval_id: str,
    ) -> None:
        ev = next(e for e in evals if e["id"] == eval_id)
        failing = failing_assertions(
            eval_runs[eval_id],
            ev["assertions"],
            handlers,
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

    if trigger == "skill":

        @pytest.mark.live_eval
        def test_should_trigger_evals_invoked_skill(
            eval_runs: dict[str, list[EvalRun]],
            live_eval_target_rate: float,
        ) -> None:
            counts = trigger_pass_counts(eval_runs, evals)
            failures = [
                (eid, n, total)
                for eid, n, total in counts
                if not eval_passed(n, total, live_eval_target_rate)
            ]
            assert not failures, (
                f"{subject_name} invoked below the bar "
                f"(P(rate >= {live_eval_target_rate:.3f}) must be >= 0.5): "
                + ", ".join(f"{eid}: {n}/{total}" for eid, n, total in failures)
            )

        trigger_test = test_should_trigger_evals_invoked_skill
    else:

        @pytest.mark.live_eval
        def test_should_invoke_agent_evals(
            eval_runs: dict[str, list[EvalRun]],
            live_eval_target_rate: float,
        ) -> None:
            counts = _agent_trigger_pass_counts(
                eval_runs, evals_path, subject_name, agent_trigger_assertion
            )
            failures = [
                (eid, n, total)
                for eid, n, total in counts
                if not eval_passed(n, total, live_eval_target_rate)
            ]
            assert not failures, (
                f"{subject_name} agent invoked below the bar "
                f"(P(rate >= {live_eval_target_rate:.3f}) must be >= 0.5): "
                + ", ".join(f"{eid}: {n}/{total}" for eid, n, total in failures)
            )

        trigger_test = test_should_invoke_agent_evals

    for fn in (test_eval_assertion, test_eval_expectation, trigger_test):
        fn.__module__ = module_name
        if isinstance(namespace, dict):
            namespace[fn.__name__] = fn
        else:
            setattr(namespace, fn.__name__, fn)
