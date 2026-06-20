"""Unit tests for `binom_eval.runner` (env scrubbing + batched runs).

`run_claude` itself spawns a real `claude -p` subprocess, so it is exercised
only through the live evals; here `run_claude_batch` is tested with the
per-call runner monkeypatched out.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import binom_eval
from binom_eval import EvalRun, stripped_env


class TestStrippedEnv:
    def test_removes_nested_session_markers(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("CLAUDECODE", "1")
        monkeypatch.setenv("CLAUDE_CODE_SESSION_ID", "abc")
        monkeypatch.setenv("CLAUDE_CODE_CHILD_SESSION", "1")
        monkeypatch.setenv("OTHER", "v")
        env = stripped_env()
        assert "CLAUDECODE" not in env
        assert "CLAUDE_CODE_SESSION_ID" not in env
        assert "CLAUDE_CODE_CHILD_SESSION" not in env
        assert env.get("OTHER") == "v"


class TestRunClaudeBatch:
    def test_runs_count_times_and_stamps_eval_id(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def fake_run(prompt: str, repo_root: Path, skill_name: str) -> EvalRun:
            return EvalRun(
                eval_id="", prompt=prompt, skill_invoked=True, assistant_text=""
            )

        monkeypatch.setattr(binom_eval.runner, "run_claude", fake_run)
        runs = binom_eval.run_claude_batch(
            {"id": "e1", "prompt": "p"}, Path("."), "demo", count=3
        )
        assert len(runs) == 3
        assert all(run.eval_id == "e1" for run in runs)
        assert all(run.prompt == "p" for run in runs)