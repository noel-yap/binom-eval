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
    _trigger_check,
    assert_check,
    eval_passed,
    expand_evals,
    failing_assertions,
    format_posterior_summary,
    graded_runs,
    graded_runs_verbose_message,
    load_evals,
    trial_outcomes,
    trial_outcomes_failure_message,
    trial_outcomes_passed,
    trial_outcomes_posterior_summary,
    trial_outcomes_verbose_message,
    trigger_pass_counts,
)
from binom_eval.plugin import (
    make_eval_runs_fixture,
    record_live_eval_posterior,
)
from binom_eval.stream_json import EvalRun, agent_invoked

TriggerMode = Literal["skill", "agent"]


def bind_eval_runs_fixture(
    eval_dir: Path,
    subject_name: str,
    handlers: dict[str, Callable[[EvalRun], None]],
    *,
    repo_root: Path | None = None,
    make_fixture: Callable[..., Any] = make_eval_runs_fixture,
) -> Callable[..., dict[str, list[EvalRun]]]:
    """Return a session-scoped ``eval_runs`` fixture for an eval directory.

    ``eval_dir`` is the directory containing ``evals.json`` (typically the
    suite's ``evals/`` folder). When ``repo_root`` is omitted, ``claude -p``
    runs with ``eval_dir`` as the working tree (the bundled example pattern);
    pass an explicit repo root when prompts reference files elsewhere.
    ``make_fixture`` defaults to `make_eval_runs_fixture` and exists so
    callers/tests can inject a different fixture factory.
    """
    eval_dir = Path(eval_dir).resolve()
    evals_path = eval_dir / "evals.json"
    root = eval_dir if repo_root is None else Path(repo_root).resolve()
    return make_fixture(evals_path, root, subject_name, handlers)


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


def _agent_trigger_check(agent_name: str) -> Callable[[EvalRun], None]:
    def check(run: EvalRun) -> None:
        sections = (("Assistant reply", run.assistant_text or "(empty)"),)
        assert_check(
            agent_invoked(run, agent_name),
            "agent was not invoked",
            sections=sections,
        )

    return check


def _record_count_posteriors(
    request: pytest.FixtureRequest,
    eval_runs: dict[str, list[EvalRun]],
    counts: list[tuple[str, int, int]],
    target: float,
    pass_threshold: float,
    *,
    verbose: bool = False,
    max_chars: int = 0,
    check: Callable[[EvalRun], None] | None = None,
) -> None:
    """Attach a posterior summary per ``(label, passes, trials)`` triple."""
    for label, passes, trials in counts:
        if verbose and check is not None:
            record_live_eval_posterior(
                request.node,
                graded_runs_verbose_message(
                    eval_runs[label],
                    label,
                    passes,
                    trials,
                    target,
                    check,
                    pass_threshold=pass_threshold,
                    max_chars=max_chars,
                ),
            )
        else:
            record_live_eval_posterior(
                request.node,
                format_posterior_summary(
                    label, passes, trials, target, pass_threshold=pass_threshold
                ),
            )


def _pass_summary(
    runs: list[EvalRun],
    outcomes: list[tuple[int, Any]],
    handler: Callable[[EvalRun], None],
    target: float,
    label: str,
    *,
    pass_threshold: float,
    max_chars: int,
    verbose: bool,
) -> str:
    """Pass-side summary: full trial detail when verbose, else one line."""
    if verbose:
        return trial_outcomes_verbose_message(
            runs,
            outcomes,
            handler,
            target,
            label,
            pass_threshold=pass_threshold,
            max_chars=max_chars,
        )
    return trial_outcomes_posterior_summary(
        outcomes, target, label, pass_threshold=pass_threshold
    )


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
        live_eval_pass_threshold: float,
        live_eval_failure_max_chars: int,
        live_eval_verbose: bool,
        live_eval_show_posterior: bool,
        request: pytest.FixtureRequest,
        eval_id: str,
        assertion_id: str,
    ) -> None:
        handler = handlers[assertion_id]
        outcomes = trial_outcomes(eval_runs[eval_id], handler)
        label = f"{eval_id}::{assertion_id}"
        passed = trial_outcomes_passed(outcomes, live_eval_target_rate)
        if passed and (live_eval_verbose or live_eval_show_posterior):
            summary = _pass_summary(
                eval_runs[eval_id],
                outcomes,
                handler,
                live_eval_target_rate,
                label,
                pass_threshold=live_eval_pass_threshold,
                max_chars=live_eval_failure_max_chars,
                verbose=live_eval_verbose,
            )
            record_live_eval_posterior(request.node, summary)
        assert passed, (
            trial_outcomes_failure_message(
                outcomes,
                live_eval_target_rate,
                label,
                max_chars=live_eval_failure_max_chars,
            )
        )

    @pytest.mark.live_eval
    @pytest.mark.parametrize("eval_id", [ev["id"] for ev in evals])
    def test_eval_expectation(
        eval_runs: dict[str, list[EvalRun]],
        live_eval_target_rate: float,
        live_eval_pass_threshold: float,
        live_eval_failure_max_chars: int,
        live_eval_verbose: bool,
        live_eval_show_posterior: bool,
        request: pytest.FixtureRequest,
        eval_id: str,
    ) -> None:
        ev = next(e for e in evals if e["id"] == eval_id)
        failing = failing_assertions(
            eval_runs[eval_id],
            ev["assertions"],
            handlers,
            live_eval_target_rate,
        )
        if not failing and (live_eval_verbose or live_eval_show_posterior):
            for assertion in ev["assertions"]:
                handler = handlers[assertion["id"]]
                outcomes = trial_outcomes(eval_runs[eval_id], handler)
                label = f"{eval_id}::{assertion['id']}"
                summary = _pass_summary(
                    eval_runs[eval_id],
                    outcomes,
                    handler,
                    live_eval_target_rate,
                    label,
                    pass_threshold=live_eval_pass_threshold,
                    max_chars=live_eval_failure_max_chars,
                    verbose=live_eval_verbose,
                )
                record_live_eval_posterior(request.node, summary)
        assert not failing, (
            f"{eval_id}: {len(failing)} assertion(s) below the bar "
            f"(P(θ ≥ {live_eval_target_rate:.3f}) must be >= 0.5):\n"
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
            live_eval_pass_threshold: float,
            live_eval_failure_max_chars: int,
            live_eval_verbose: bool,
            live_eval_show_posterior: bool,
            request: pytest.FixtureRequest,
        ) -> None:
            counts = trigger_pass_counts(eval_runs, evals)
            failures = [
                (eid, n, total)
                for eid, n, total in counts
                if not eval_passed(n, total, live_eval_target_rate)
            ]
            if not failures and (live_eval_verbose or live_eval_show_posterior):
                _record_count_posteriors(
                    request,
                    eval_runs,
                    counts,
                    live_eval_target_rate,
                    live_eval_pass_threshold,
                    verbose=live_eval_verbose,
                    max_chars=live_eval_failure_max_chars,
                    check=_trigger_check,
                )
            assert not failures, (
                f"{subject_name} invoked below the bar "
                f"(P(θ ≥ {live_eval_target_rate:.3f}) must be >= 0.5): "
                + ", ".join(f"{eid}: {n}/{total}" for eid, n, total in failures)
            )

        trigger_test = test_should_trigger_evals_invoked_skill
    else:
        agent_check = _agent_trigger_check(subject_name)

        @pytest.mark.live_eval
        def test_should_invoke_agent_evals(
            eval_runs: dict[str, list[EvalRun]],
            live_eval_target_rate: float,
            live_eval_pass_threshold: float,
            live_eval_failure_max_chars: int,
            live_eval_verbose: bool,
            live_eval_show_posterior: bool,
            request: pytest.FixtureRequest,
        ) -> None:
            counts = _agent_trigger_pass_counts(
                eval_runs, evals_path, subject_name, agent_trigger_assertion
            )
            failures = [
                (eid, n, total)
                for eid, n, total in counts
                if not eval_passed(n, total, live_eval_target_rate)
            ]
            if not failures and (live_eval_verbose or live_eval_show_posterior):
                _record_count_posteriors(
                    request,
                    eval_runs,
                    counts,
                    live_eval_target_rate,
                    live_eval_pass_threshold,
                    verbose=live_eval_verbose,
                    max_chars=live_eval_failure_max_chars,
                    check=agent_check,
                )
            assert not failures, (
                f"{subject_name} agent invoked below the bar "
                f"(P(θ ≥ {live_eval_target_rate:.3f}) must be >= 0.5): "
                + ", ".join(f"{eid}: {n}/{total}" for eid, n, total in failures)
            )

        trigger_test = test_should_invoke_agent_evals

    for fn in (test_eval_assertion, test_eval_expectation, trigger_test):
        fn.__module__ = module_name
        if isinstance(namespace, dict):
            namespace[fn.__name__] = fn
        else:
            setattr(namespace, fn.__name__, fn)
