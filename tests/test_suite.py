"""Unit tests for ``binom_eval.suite`` consumer helpers."""

from __future__ import annotations

from pathlib import Path

import pytest

from binom_eval import AssertionFailure, EvalRun, bind_eval_runs_fixture, register_live_eval_tests


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
