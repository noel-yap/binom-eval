"""Unit tests for ``binom_eval.suite`` consumer helpers."""

from __future__ import annotations

from pathlib import Path

import pytest

from binom_eval import (
    AssertionFailure,
    EvalRun,
    PASS_THRESHOLD,
    assert_check,
    bind_eval_runs_fixture,
    register_live_eval_tests,
    trial_outcomes,
)
from binom_eval.grading import _trigger_check
from binom_eval.plugin import LIVE_EVAL_POSTERIOR_PROPERTY
from binom_eval.suite import _pass_summary, _record_count_posteriors


def test_bind_eval_runs_fixture_delegates_to_make_eval_runs_fixture(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    eval_dir = Path("/tmp/skill/evals")
    handlers: dict = {"a": lambda r: None}
    sentinel = object()
    captured: dict = {}

    def fake_make(*args, **kwargs):
        captured["args"] = args
        return sentinel

    monkeypatch.setattr("binom_eval.suite.make_eval_runs_fixture", fake_make)
    result = bind_eval_runs_fixture(eval_dir, "my-skill", handlers)
    assert result is sentinel
    assert captured["args"] == (
        eval_dir.resolve() / "evals.json",
        eval_dir.resolve(),
        "my-skill",
        handlers,
    )


def test_bind_eval_runs_fixture_accepts_explicit_repo_root(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    eval_dir = Path("/tmp/skill/evals")
    repo_root = Path("/tmp/repo")
    handlers: dict = {}
    captured: dict = {}

    monkeypatch.setattr(
        "binom_eval.suite.make_eval_runs_fixture",
        lambda *args, **kwargs: captured.setdefault("args", args),
    )
    bind_eval_runs_fixture(eval_dir, "my-skill", handlers, repo_root=repo_root)
    assert captured["args"][1] == repo_root.resolve()


def test_register_live_eval_tests_attaches_skill_tests(
    tmp_path: Path,
) -> None:
    evals_path = tmp_path / "evals.json"
    evals_path.write_text(
        """{
  "evals": [
    {
      "id": "positive",
      "should_trigger": true,
      "prompt": "go",
      "expected_output": "did the thing",
      "assertions": [{"id": "checks-out"}]
    }
  ]
}""",
        encoding="utf-8",
    )

    def _checks_out(run: EvalRun) -> None:
        if "ok" not in run.assistant_text:
            raise AssertionFailure("missing ok")

    handlers = {"checks-out": _checks_out}
    namespace: dict[str, object] = {"__name__": "fake_skill_evals"}
    register_live_eval_tests(
        namespace,
        evals_path=evals_path,
        handlers=handlers,
        subject_name="example-skill",
        trigger="skill",
    )

    assert "test_eval_assertion" in namespace
    assert "test_eval_expectation" in namespace
    assert "test_should_trigger_evals_invoked_skill" in namespace
    assert namespace["test_eval_assertion"].__module__ == "fake_skill_evals"  # type: ignore[union-attr]


def test_register_live_eval_tests_attaches_agent_tests(
    tmp_path: Path,
) -> None:
    evals_path = tmp_path / "evals.json"
    evals_path.write_text(
        """{
  "evals": [
    {
      "id": "positive",
      "prompt": "go",
      "expected_output": "delegated",
      "assertions": [{"id": "invokes-agent"}]
    }
  ]
}""",
        encoding="utf-8",
    )

    handlers = {"invokes-agent": lambda r: None}
    namespace: dict[str, object] = {"__name__": "fake_agent_evals"}
    register_live_eval_tests(
        namespace,
        evals_path=evals_path,
        handlers=handlers,
        subject_name="my-agent",
        trigger="agent",
    )

    assert "test_should_invoke_agent_evals" in namespace
    assert "test_should_trigger_evals_invoked_skill" not in namespace


class _StubNode:
    def __init__(self) -> None:
        self.user_properties: list[tuple[str, str]] = []


class _StubRequest:
    def __init__(self) -> None:
        self.node = _StubNode()


def _demo_namespace(
    tmp_path: Path, handlers: dict
) -> dict[str, object]:
    """Register the standard tests for a one-eval suite; return namespace."""
    evals_path = tmp_path / "evals.json"
    evals_path.write_text(
        """{
  "evals": [
    {
      "id": "positive",
      "should_trigger": true,
      "prompt": "go",
      "expected_output": "did the thing",
      "assertions": [{"id": "checks-out"}]
    }
  ]
}""",
        encoding="utf-8",
    )
    namespace: dict[str, object] = {"__name__": "fake_skill_evals"}
    register_live_eval_tests(
        namespace,
        evals_path=evals_path,
        handlers=handlers,
        subject_name="example-skill",
        trigger="skill",
    )
    return namespace


def _passing_runs() -> dict[str, list[EvalRun]]:
    return {
        "positive": [
            EvalRun(
                eval_id="positive",
                prompt="go",
                skill_invoked=True,
                assistant_text="ok",
            )
            for _ in range(3)
        ]
    }


def test_record_count_posteriors_attaches_one_summary_per_count() -> None:
    request = _StubRequest()

    _record_count_posteriors(
        request,
        {},
        [("e1", 3, 3), ("e2", 2, 3)],
        2.0 / 3.0,
        PASS_THRESHOLD,
    )

    labels = [value for _, value in request.node.user_properties]
    names = [name for name, _ in request.node.user_properties]
    assert names == [LIVE_EVAL_POSTERIOR_PROPERTY] * 2
    assert labels[0].startswith("e1: 3/3 trials passed;")
    assert "max θ₀ (pass@τ=" in labels[0]
    assert labels[1].startswith("e2: 2/3 trials passed;")


def test_record_count_posteriors_verbose_includes_trial_detail() -> None:
    request = _StubRequest()
    runs = {
        "positive": [
            EvalRun(
                eval_id="positive",
                prompt="go",
                skill_invoked=True,
                assistant_text="hello",
            )
        ]
    }

    _record_count_posteriors(
        request,
        runs,
        [("positive", 1, 1)],
        2.0 / 3.0,
        PASS_THRESHOLD,
        verbose=True,
        max_chars=2000,
        check=_trigger_check,
    )

    _, summary = request.node.user_properties[0]
    assert "Trials:" in summary
    assert "trial 0: passed" in summary
    assert "Assistant reply:" in summary
    assert "hello" in summary


def test_record_count_posteriors_with_no_counts_records_nothing() -> None:
    request = _StubRequest()

    _record_count_posteriors(request, {}, [], 2.0 / 3.0, PASS_THRESHOLD)

    assert request.node.user_properties == []


def test_pass_summary_verbose_returns_full_trial_detail() -> None:
    runs = [
        EvalRun(
            eval_id="e",
            prompt="go",
            skill_invoked=True,
            assistant_text="ok",
        )
    ]

    def check(run: EvalRun) -> None:
        assert_check(run.skill_invoked, "miss")

    outcomes = trial_outcomes(runs, check)

    summary = _pass_summary(
        runs,
        outcomes,
        check,
        2.0 / 3.0,
        "e::check",
        pass_threshold=PASS_THRESHOLD,
        max_chars=2000,
        verbose=True,
    )

    assert "Trials:" in summary
    assert "trial 0: passed" in summary


def test_pass_summary_not_verbose_returns_one_line_posterior_summary() -> None:
    runs = [
        EvalRun(
            eval_id="e",
            prompt="go",
            skill_invoked=True,
            assistant_text="ok",
        )
    ]

    def check(run: EvalRun) -> None:
        assert_check(run.skill_invoked, "miss")

    outcomes = trial_outcomes(runs, check)

    summary = _pass_summary(
        runs,
        outcomes,
        check,
        2.0 / 3.0,
        "e::check",
        pass_threshold=PASS_THRESHOLD,
        max_chars=2000,
        verbose=False,
    )

    assert summary.startswith("e::check: 1/1 trials passed;")
    assert "Trials:" not in summary


def test_assertion_test_records_verbose_detail_when_enabled(
    tmp_path: Path,
) -> None:
    def _checks_out(run: EvalRun) -> None:
        sections = (
            ("Input", run.prompt),
            ("Output", run.assistant_text),
        )
        assert_check(True, "missing ok", sections=sections)

    namespace = _demo_namespace(tmp_path, {"checks-out": _checks_out})
    request = _StubRequest()

    namespace["test_eval_assertion"](
        eval_runs=_passing_runs(),
        live_eval_target_rate=2.0 / 3.0,
        live_eval_pass_threshold=PASS_THRESHOLD,
        live_eval_failure_max_chars=2000,
        live_eval_verbose=True,
        live_eval_show_posterior=False,
        request=request,
        eval_id="positive",
        assertion_id="checks-out",
    )

    assert len(request.node.user_properties) == 1
    name, summary = request.node.user_properties[0]
    assert name == LIVE_EVAL_POSTERIOR_PROPERTY
    assert summary.startswith("positive::checks-out: 3/3 trials passed;")
    assert "Trials:" in summary
    assert "trial 0: passed" in summary
    assert "Input:" in summary
    assert "Output:" in summary
    assert "go" in summary
    assert "ok" in summary


def test_assertion_test_records_posterior_only_when_show_posterior_enabled(
    tmp_path: Path,
) -> None:
    namespace = _demo_namespace(tmp_path, {"checks-out": lambda r: None})
    request = _StubRequest()

    namespace["test_eval_assertion"](
        eval_runs=_passing_runs(),
        live_eval_target_rate=2.0 / 3.0,
        live_eval_pass_threshold=PASS_THRESHOLD,
        live_eval_failure_max_chars=2000,
        live_eval_verbose=False,
        live_eval_show_posterior=True,
        request=request,
        eval_id="positive",
        assertion_id="checks-out",
    )

    assert len(request.node.user_properties) == 1
    _, summary = request.node.user_properties[0]
    assert summary.startswith("positive::checks-out: 3/3 trials passed;")
    assert "Trials:" not in summary


def test_assertion_test_records_nothing_when_disabled(
    tmp_path: Path,
) -> None:
    namespace = _demo_namespace(tmp_path, {"checks-out": lambda r: None})
    request = _StubRequest()

    namespace["test_eval_assertion"](
        eval_runs=_passing_runs(),
        live_eval_target_rate=2.0 / 3.0,
        live_eval_pass_threshold=PASS_THRESHOLD,
        live_eval_failure_max_chars=2000,
        live_eval_verbose=False,
        live_eval_show_posterior=False,
        request=request,
        eval_id="positive",
        assertion_id="checks-out",
    )

    assert request.node.user_properties == []


def test_assertion_test_records_nothing_when_failing(
    tmp_path: Path,
) -> None:
    def _always_fails(run: EvalRun) -> None:
        raise AssertionFailure("nope")

    namespace = _demo_namespace(tmp_path, {"checks-out": _always_fails})
    request = _StubRequest()

    with pytest.raises(AssertionError):
        namespace["test_eval_assertion"](
            eval_runs=_passing_runs(),
            live_eval_target_rate=2.0 / 3.0,
            live_eval_pass_threshold=PASS_THRESHOLD,
            live_eval_failure_max_chars=2000,
            live_eval_verbose=True,
            live_eval_show_posterior=False,
            request=request,
            eval_id="positive",
            assertion_id="checks-out",
        )

    assert request.node.user_properties == []


def test_expectation_test_records_posterior_per_assertion(
    tmp_path: Path,
) -> None:
    namespace = _demo_namespace(tmp_path, {"checks-out": lambda r: None})
    request = _StubRequest()

    namespace["test_eval_expectation"](
        eval_runs=_passing_runs(),
        live_eval_target_rate=2.0 / 3.0,
        live_eval_pass_threshold=PASS_THRESHOLD,
        live_eval_failure_max_chars=2000,
        live_eval_verbose=True,
        live_eval_show_posterior=False,
        request=request,
        eval_id="positive",
    )

    assert len(request.node.user_properties) == 1
    _, summary = request.node.user_properties[0]
    assert summary.startswith("positive::checks-out: 3/3 trials passed;")
    assert "Trials:" in summary


def test_trigger_test_records_posterior_per_eval(
    tmp_path: Path,
) -> None:
    namespace = _demo_namespace(tmp_path, {"checks-out": lambda r: None})
    request = _StubRequest()

    namespace["test_should_trigger_evals_invoked_skill"](
        eval_runs=_passing_runs(),
        live_eval_target_rate=2.0 / 3.0,
        live_eval_pass_threshold=PASS_THRESHOLD,
        live_eval_failure_max_chars=2000,
        live_eval_verbose=True,
        live_eval_show_posterior=False,
        request=request,
    )

    assert len(request.node.user_properties) == 1
    _, summary = request.node.user_properties[0]
    assert summary.startswith("positive: 3/3 trials passed;")
    assert "Trials:" in summary
    assert "trial 0: passed" in summary
    assert "Assistant reply:" in summary
