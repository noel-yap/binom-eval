"""Unit tests for the `binom_eval.runner` package layer.

Covers the backend-agnostic pieces that live in the package root: env
scrubbing (`stripped_env`), the per-run workdir (`isolated_workdir`), the
pure model-probe parser (`_model_probe_rejected`), the `backend:model` spec
parser (`resolve_runner`), and the concurrent `run_claude_batch` driver. The
`ClaudeRunner` backend itself is tested in `test_claude_runner.py`;
`run_claude_batch` is exercised here against an injected fake `Runner`.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any

import pytest

import binom_eval
from binom_eval import (
    ClaudeRunner,
    CursorRunner,
    EvalRun,
    Runner,
    isolated_workdir,
    resolve_runner,
    stripped_env,
)


class _FakeRunner(Runner):
    """A `Runner` whose `run` delegates to an injected callable.

    Lets `run_claude_batch` be exercised against a backend that records or
    throttles calls without spawning any CLI; `version`/`preflight`/
    `validate_model` are unused by the batch driver and stubbed inert.
    """

    def __init__(self, run_fn: Any) -> None:
        self._run_fn = run_fn

    def version(self) -> str:
        return ""

    def preflight(self) -> str | None:
        return None

    def validate_model(self, model: str, timeout: int = 30) -> str | None:
        return None

    def run(
        self,
        prompt: str,
        repo_root: Path,
        skill_name: str,
        timeout: int = 300,
        *,
        isolate: bool = False,
        model: str,
    ) -> EvalRun:
        return self._run_fn(
            prompt, repo_root, skill_name, isolate=isolate, model=model
        )


class TestResolveRunner:
    def test_bare_model_without_prefix_raises(self) -> None:
        with pytest.raises(ValueError, match="must be 'backend:model'"):
            resolve_runner("haiku")

    def test_missing_spec_raises(self) -> None:
        with pytest.raises(ValueError, match="must be 'backend:model'"):
            resolve_runner(None)

    def test_claude_prefix_strips_to_model(self) -> None:
        backend, model, runner = resolve_runner("claude:claude-opus-4-8")
        assert backend == "claude"
        assert model == "claude-opus-4-8"
        assert isinstance(runner, ClaudeRunner)

    def test_cursor_prefix_selects_cursor_backend(self) -> None:
        backend, model, runner = resolve_runner("cursor:sonnet-4.5")
        assert backend == "cursor"
        assert model == "sonnet-4.5"
        assert isinstance(runner, CursorRunner)

    def test_unknown_backend_prefix_raises(self) -> None:
        with pytest.raises(ValueError, match="unknown eval backend 'gpt'"):
            resolve_runner("gpt:4o")

    def test_empty_model_raises(self) -> None:
        with pytest.raises(ValueError, match="empty model"):
            resolve_runner("claude:")


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
    def test_runs_count_times_and_stamps_eval_id(self) -> None:
        def fake_run(
            prompt: str,
            repo_root: Path,
            skill_name: str,
            *,
            isolate: bool = False,
            model: str,
        ) -> EvalRun:
            return EvalRun(
                eval_id="", prompt=prompt, skill_invoked=True, assistant_text=""
            )

        runs = binom_eval.run_claude_batch(
            {"id": "e1", "prompt": "p"},
            Path("."),
            "demo",
            count=3,
            model="m",
            runner=_FakeRunner(fake_run),
        )
        assert len(runs) == 3
        assert all(run.eval_id == "e1" for run in runs)
        assert all(run.prompt == "p" for run in runs)

    def test_gate_caps_concurrent_runs(self) -> None:
        lock = threading.Lock()
        live = {"now": 0, "max": 0}

        def fake_run(
            prompt: str,
            repo_root: Path,
            skill_name: str,
            *,
            isolate: bool = False,
            model: str,
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

        runs = binom_eval.run_claude_batch(
            {"id": "e1", "prompt": "p"},
            Path("."),
            "demo",
            count=6,
            gate=threading.Semaphore(2),
            model="m",
            runner=_FakeRunner(fake_run),
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


class TestModelProbeRejected:
    """Unit tests for the pure probe-parser behind `validate_model`."""

    # Trimmed stream-json from `claude -p --model <bad> --output-format
    # stream-json`: a synthetic assistant turn plus an is_error 404 result.
    _BAD = (
        '{"type":"assistant","message":{"model":"<synthetic>"},'
        '"error":"model_not_found"}\n'
        '{"type":"result","subtype":"success","is_error":true,'
        '"api_error_status":404,"result":"There\'s an issue with the '
        'selected model (nope). It may not exist."}\n'
    )
    _GOOD = (
        '{"type":"assistant","message":{"model":"claude-haiku-4-5"}}\n'
        '{"type":"result","subtype":"success","is_error":false,'
        '"result":"ok"}\n'
    )

    def test_rejects_unknown_model_with_cli_message(self) -> None:
        from binom_eval.runner import _model_probe_rejected

        msg = _model_probe_rejected(self._BAD)
        assert msg is not None
        assert "may not exist" in msg

    def test_accepts_usable_model(self) -> None:
        from binom_eval.runner import _model_probe_rejected

        assert _model_probe_rejected(self._GOOD) is None

    def test_ignores_blank_and_unparsable_lines(self) -> None:
        from binom_eval.runner import _model_probe_rejected

        assert _model_probe_rejected("\n  \nnot json\n") is None

    # --- additional probe-parser scenarios (independent conditions) ---
    # An is_error result carrying HTTP 404 but no `model_not_found` marker.
    _BAD_404_ONLY = (
        '{"type":"assistant","message":{"model":"<synthetic>"}}\n'
        '{"type":"result","subtype":"success","is_error":true,'
        '"api_error_status":404,"result":"model unavailable"}\n'
    )
    # A `model_not_found` marker with no result line carrying a message.
    _BAD_NO_MESSAGE = (
        '{"type":"assistant","message":{"model":"<synthetic>"},'
        '"error":"model_not_found"}\n'
    )
    # Real "Not logged in" output: is_error true, but it is an auth failure
    # (api_error_status null, no model_not_found), so the model is not at fault.
    _AUTH_FAIL = (
        '{"type":"assistant","message":{"model":"<synthetic>"},'
        '"error":"authentication_failed"}\n'
        '{"type":"result","subtype":"success","is_error":true,'
        '"api_error_status":null,"result":"Not logged in"}\n'
    )
    # A 404 that appears on a non-result line must be ignored.
    _IS_ERROR_NOT_RESULT = (
        '{"type":"assistant","is_error":true,"api_error_status":404,'
        '"message":{"model":"x"}}\n'
        '{"type":"result","subtype":"success","is_error":false,"result":"ok"}\n'
    )

    def test_rejects_on_http_404_without_model_not_found_marker(self) -> None:
        from binom_eval.runner import _model_probe_rejected

        assert _model_probe_rejected(self._BAD_404_ONLY) == "model unavailable"

    def test_rejects_with_default_message_when_no_result_text(self) -> None:
        from binom_eval.runner import _model_probe_rejected

        assert _model_probe_rejected(self._BAD_NO_MESSAGE) == "model not found"

    def test_accepts_when_error_is_auth_not_model(self) -> None:
        from binom_eval.runner import _model_probe_rejected

        # is_error is true, but it is an auth failure (no 404, no
        # model_not_found), so the probe must not blame the model.
        assert _model_probe_rejected(self._AUTH_FAIL) is None

    def test_ignores_http_404_on_a_non_result_event(self) -> None:
        from binom_eval.runner import _model_probe_rejected

        assert _model_probe_rejected(self._IS_ERROR_NOT_RESULT) is None
