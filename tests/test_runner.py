"""Unit tests for `binom_eval.runner` (env scrubbing + batched runs).

`run_claude` itself spawns a real `claude -p` subprocess, so it is exercised
only through the live evals; here `run_claude_batch` is tested with the
per-call runner monkeypatched out.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest

import binom_eval
from binom_eval import EvalRun, isolated_workdir, stripped_env


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



class TestCliVersion:
    def test_returns_string(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import subprocess

        monkeypatch.setattr(
            subprocess,
            'run',
            lambda *a, **kw: type('R', (), {'stdout': '1.2.3\n'})(),
        )
        from binom_eval.runner import cli_version

        assert cli_version() == '1.2.3'

    def test_returns_empty_when_not_found(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import subprocess

        def raise_fnf(*a: object, **kw: object) -> None:
            raise FileNotFoundError

        monkeypatch.setattr(subprocess, 'run', raise_fnf)
        from binom_eval.runner import cli_version

        assert cli_version() == ''

    def test_returns_empty_on_timeout(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import subprocess

        def raise_timeout(*a: object, **kw: object) -> None:
            raise subprocess.TimeoutExpired(cmd='claude', timeout=10)

        monkeypatch.setattr(subprocess, 'run', raise_timeout)
        from binom_eval.runner import cli_version

        assert cli_version() == ''


class TestRunClaudeBatch:
    def test_runs_count_times_and_stamps_eval_id(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def fake_run(
            prompt: str,
            repo_root: Path,
            skill_name: str,
            *,
            isolate: bool = False,
            model: str | None = None,
        ) -> EvalRun:
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

    def test_gate_caps_concurrent_runs(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        lock = threading.Lock()
        live = {"now": 0, "max": 0}

        def fake_run(
            prompt: str,
            repo_root: Path,
            skill_name: str,
            *,
            isolate: bool = False,
            model: str | None = None,
        ) -> EvalRun:
            with lock:
                live["now"] += 1
                live["max"] = max(live["max"], live["now"])
            # A brief overlap window so unthrottled runs would pile up; the
            # gate must still hold the peak at its count.
            time.sleep(0.02)
            with lock:
                live["now"] -= 1
            return EvalRun(
                eval_id="", prompt=prompt, skill_invoked=True, assistant_text=""
            )

        monkeypatch.setattr(binom_eval.runner, "run_claude", fake_run)
        runs = binom_eval.run_claude_batch(
            {"id": "e1", "prompt": "p"},
            Path("."),
            "demo",
            count=6,
            gate=threading.Semaphore(2),
        )
        assert len(runs) == 6
        assert live["max"] <= 2


class TestIsolatedWorkdir:
    def test_without_isolation_yields_repo_root_unchanged(
        self, tmp_path: Path
    ) -> None:
        with isolated_workdir(tmp_path, isolate=False) as workdir:
            assert workdir == tmp_path

    def test_isolation_copies_tree_skips_ignored_and_cleans_up(
        self, tmp_path: Path
    ) -> None:
        (tmp_path / "keep.txt").write_text("payload", encoding="utf-8")
        cache = tmp_path / "__pycache__"
        cache.mkdir()
        (cache / "junk.pyc").write_text("nope", encoding="utf-8")

        with isolated_workdir(tmp_path, isolate=True) as workdir:
            assert workdir != tmp_path
            assert (workdir / "keep.txt").read_text(encoding="utf-8") == (
                "payload"
            )
            assert not (workdir / "__pycache__").exists()
            copied = workdir

        # The throwaway copy is removed once the run ends, and the original
        # tree is untouched.
        assert not copied.exists()
        assert (tmp_path / "keep.txt").exists()
