"""Unit tests for `binom_eval.runner.claude_runner.ClaudeRunner`.

`ClaudeRunner.run` spawns a real `claude -p` subprocess, so its full path is
exercised by the live evals; here the subprocess and the stream-json parser
are stubbed so the command-assembly and result-wiring branches can be
checked without invoking the CLI. `version` and `validate_model` stub only
`subprocess.run`.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

import pytest

from binom_eval import EvalRun
from binom_eval.runner import claude_runner
from binom_eval.runner.claude_runner import ClaudeRunner


class TestClaudeRunnerVersion:
    def test_returns_stripped_version_on_success(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            subprocess,
            "run",
            lambda *a, **kw: type("R", (), {"stdout": "1.2.3\n"})(),
        )
        assert ClaudeRunner().version() == "1.2.3"

    def test_returns_empty_when_cli_absent(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def raise_fnf(*a: object, **kw: object) -> None:
            raise FileNotFoundError

        monkeypatch.setattr(subprocess, "run", raise_fnf)
        assert ClaudeRunner().version() == ""

    def test_returns_empty_on_timeout(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def raise_timeout(*a: object, **kw: object) -> None:
            raise subprocess.TimeoutExpired(cmd="claude", timeout=10)

        monkeypatch.setattr(subprocess, "run", raise_timeout)
        assert ClaudeRunner().version() == ""


class TestClaudeRunnerValidateModel:
    # Trimmed stream-json from a `--model <bad>` probe: an is_error 404 result.
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

    def test_returns_none_when_cli_absent(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def raise_fnf(*a: object, **kw: object) -> None:
            raise FileNotFoundError

        monkeypatch.setattr(subprocess, "run", raise_fnf)
        assert ClaudeRunner().validate_model("anything") is None

    def test_returns_none_on_timeout(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def raise_timeout(*a: object, **kw: object) -> None:
            raise subprocess.TimeoutExpired(cmd="claude", timeout=1)

        monkeypatch.setattr(subprocess, "run", raise_timeout)
        assert ClaudeRunner().validate_model("haiku") is None

    def test_reports_bad_model_from_stdout(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            subprocess,
            "run",
            lambda *a, **kw: type("R", (), {"stdout": self._BAD})(),
        )
        msg = ClaudeRunner().validate_model("bad")
        assert msg is not None and "may not exist" in msg

    def test_accepts_good_model_from_stdout(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            subprocess,
            "run",
            lambda *a, **kw: type("R", (), {"stdout": self._GOOD})(),
        )
        assert ClaudeRunner().validate_model("haiku") is None


class TestClaudeRunnerPreflight:
    def test_ready_when_cli_present_and_key_set(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            claude_runner.shutil, "which", lambda _n: "/usr/bin/claude"
        )
        monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
        assert ClaudeRunner().preflight() is None

    def test_reports_missing_cli(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(claude_runner.shutil, "which", lambda _n: None)
        monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
        msg = ClaudeRunner().preflight()
        assert msg is not None and "claude CLI not found" in msg

    def test_reports_missing_api_key(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            claude_runner.shutil, "which", lambda _n: "/usr/bin/claude"
        )
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        msg = ClaudeRunner().preflight()
        assert msg is not None and "ANTHROPIC_API_KEY" in msg


class TestClaudeRunnerRun:
    @staticmethod
    def _stub_subprocess(
        monkeypatch: pytest.MonkeyPatch, captured: dict[str, Any]
    ) -> None:
        """Capture the `subprocess.run` call and return an empty fake proc."""

        def fake_run(cmd: list[str], **kw: Any) -> Any:
            captured["cmd"] = cmd
            captured["cwd"] = kw.get("cwd")
            return type("R", (), {"stdout": ""})()

        monkeypatch.setattr(subprocess, "run", fake_run)
        monkeypatch.setattr(
            claude_runner,
            "parse_stream_json",
            lambda _stdout, _skill: (True, "answer", [], "resolved-model"),
        )

    def test_includes_model_flag_when_model_given(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        captured: dict[str, Any] = {}
        self._stub_subprocess(monkeypatch, captured)

        run = ClaudeRunner().run(
            "do it", tmp_path, "demo", model="claude-haiku-4-5"
        )

        cmd = captured["cmd"]
        assert "--model" in cmd
        assert cmd[cmd.index("--model") + 1] == "claude-haiku-4-5"
        assert isinstance(run, EvalRun)
        assert run.skill_invoked is True
        assert run.assistant_text == "answer"
        assert run.model == "resolved-model"

    def test_always_includes_model_flag(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        captured: dict[str, Any] = {}
        self._stub_subprocess(monkeypatch, captured)

        ClaudeRunner().run("do it", tmp_path, "demo", model="claude-haiku-4-5")

        assert "--model" in captured["cmd"]

    def test_runs_in_repo_root_without_isolation(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        captured: dict[str, Any] = {}
        self._stub_subprocess(monkeypatch, captured)

        ClaudeRunner().run(
            "do it", tmp_path, "demo", isolate=False, model="claude-haiku-4-5"
        )

        assert captured["cwd"] == str(tmp_path)
