"""Unit tests for `binom_eval.runner.claude_runner.ClaudeRunner`.

`ClaudeRunner.run` spawns a real `claude -p` subprocess, so its full path is
exercised by the live evals; here the subprocess and the stream-json parser
are stubbed so the command-assembly and result-wiring branches can be
checked without invoking the CLI. `version` and `validate_model` stub only
`subprocess.run`.
"""

from __future__ import annotations

import io
import json
import subprocess
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import pytest

from binom_eval import EvalRun
from binom_eval.runner import TRIAL_RETRY, claude_runner
from binom_eval.runner import retry as runner_retry
from binom_eval.runner.claude_runner import ClaudeRunner

# A minimal clean trial stream: one assistant event and a non-error result,
# so `_run_error` grades the run as completed.
_OK_STREAM = (
    '{"type":"assistant","message":{"content":[{"type":"text","text":"hi"}]}}\n'
    '{"type":"result","subtype":"success","is_error":false,"result":"ok"}\n'
)

# A trial whose result event reports an execution error (e.g. API 500).
_ERROR_STREAM = (
    '{"type":"assistant","message":{"content":[]}}\n'
    '{"type":"result","subtype":"error_during_execution","is_error":true,'
    '"result":"API Error: 500"}\n'
)


def _proc(
    stdout: str = _OK_STREAM, returncode: int = 0, stderr: str = ""
) -> Any:
    """A fake completed subprocess with the attributes `run` inspects."""
    return type(
        "R",
        (),
        {"stdout": stdout, "returncode": returncode, "stderr": stderr},
    )()


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
    _LISTED = ["claude-haiku-4-5-20251001", "claude-opus-4-8-20250101"]

    @staticmethod
    def _stub_api_list(
        monkeypatch: pytest.MonkeyPatch, ids: list[str] | None
    ) -> None:
        monkeypatch.setattr(
            claude_runner, "_anthropic_model_ids", lambda _timeout=30: ids
        )

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

    def test_accepts_exact_model_id_from_api_list(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._stub_api_list(monkeypatch, self._LISTED)
        called: list[object] = []
        monkeypatch.setattr(
            subprocess,
            "run",
            lambda *a, **kw: called.append(a) or type("R", (), {"stdout": ""})(),
        )
        assert (
            ClaudeRunner().validate_model("claude-haiku-4-5-20251001") is None
        )
        assert called == []

    def test_accepts_model_prefix_from_api_list(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._stub_api_list(monkeypatch, self._LISTED)
        assert ClaudeRunner().validate_model("claude-opus-4-8") is None

    def test_accepts_cli_alias_from_api_list(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._stub_api_list(monkeypatch, self._LISTED)
        assert ClaudeRunner().validate_model("haiku") is None

    def test_reports_bad_model_from_api_list(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._stub_api_list(monkeypatch, self._LISTED)
        called: list[object] = []
        monkeypatch.setattr(
            subprocess,
            "run",
            lambda *a, **kw: called.append(a) or type("R", (), {"stdout": ""})(),
        )
        msg = ClaudeRunner().validate_model("nope")
        assert msg is not None and "model not found: nope" in msg
        assert (
            "valid models: claude-haiku-4-5-20251001, claude-opus-4-8-20250101"
            in msg
        )
        assert called == []

    def test_returns_none_when_cli_absent(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._stub_api_list(monkeypatch, None)

        def raise_fnf(*a: object, **kw: object) -> None:
            raise FileNotFoundError

        monkeypatch.setattr(subprocess, "run", raise_fnf)
        assert ClaudeRunner().validate_model("anything") is None

    def test_returns_none_on_timeout(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._stub_api_list(monkeypatch, None)

        def raise_timeout(*a: object, **kw: object) -> None:
            raise subprocess.TimeoutExpired(cmd="claude", timeout=1)

        monkeypatch.setattr(subprocess, "run", raise_timeout)
        assert ClaudeRunner().validate_model("haiku") is None

    def test_falls_back_to_probe_when_api_list_unavailable(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._stub_api_list(monkeypatch, None)
        monkeypatch.setattr(
            subprocess,
            "run",
            lambda *a, **kw: type("R", (), {"stdout": self._BAD})(),
        )
        msg = ClaudeRunner().validate_model("bad")
        assert msg is not None and "may not exist" in msg
        assert "valid models" not in msg

    def test_accepts_good_model_from_probe_when_api_list_unavailable(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._stub_api_list(monkeypatch, None)
        monkeypatch.setattr(
            subprocess,
            "run",
            lambda *a, **kw: type("R", (), {"stdout": self._GOOD})(),
        )
        assert ClaudeRunner().validate_model("haiku") is None


class TestModelInAnthropicList:
    def test_matches_exact_id(self) -> None:
        ids = ["claude-haiku-4-5-20251001"]
        assert claude_runner._model_in_anthropic_list(
            "claude-haiku-4-5-20251001", ids
        )

    def test_matches_id_prefix(self) -> None:
        ids = ["claude-opus-4-8-20250101"]
        assert claude_runner._model_in_anthropic_list("claude-opus-4-8", ids)

    def test_matches_cli_alias(self) -> None:
        assert claude_runner._model_in_anthropic_list("sonnet", [])

    def test_rejects_unknown_model(self) -> None:
        ids = ["claude-haiku-4-5-20251001"]
        assert not claude_runner._model_in_anthropic_list("nope", ids)


class TestAnthropicModelsApi:
    _PAYLOAD = {"data": [{"id": "claude-haiku-4-5-20251001"}]}

    @staticmethod
    def _http_error(code: int) -> urllib.error.HTTPError:
        return urllib.error.HTTPError(
            "https://api.anthropic.com/v1/models?limit=100",
            code,
            "err",
            hdrs=None,
            fp=io.BytesIO(b""),
        )

    @staticmethod
    def _ok_response() -> urllib.request.addinfourl:
        body = json.dumps(TestAnthropicModelsApi._PAYLOAD).encode()
        return urllib.request.addinfourl(io.BytesIO(body), {}, url="")

    def test_backoff_delay_is_jittered_within_cap(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(runner_retry.random, "uniform", lambda _a, b: b)
        assert claude_runner._MODELS_API_RETRY.backoff_delay(0) == 0.25
        assert claude_runner._MODELS_API_RETRY.backoff_delay(1) == 0.5
        assert claude_runner._MODELS_API_RETRY.backoff_delay(10) == 2.0

    def test_transient_http_code_raises_retryable(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def fake_urlopen(_req: object, timeout: float) -> object:
            raise self._http_error(503)

        monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
        with pytest.raises(runner_retry.RetryableError):
            claude_runner._fetch_anthropic_model_ids_once(object(), 5.0)

    def test_non_transient_http_code_returns_none(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def fake_urlopen(_req: object, timeout: float) -> object:
            raise self._http_error(401)

        monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
        assert claude_runner._fetch_anthropic_model_ids_once(object(), 5.0) is None

    def test_retries_transient_error_then_succeeds(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls = {"n": 0}

        def fake_urlopen(_req: object, timeout: float) -> object:
            calls["n"] += 1
            if calls["n"] < 3:
                raise self._http_error(503)
            return self._ok_response()

        monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
        monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
        monkeypatch.setattr(runner_retry.time, "sleep", lambda _delay: None)

        ids = claude_runner._anthropic_model_ids()
        assert ids == ["claude-haiku-4-5-20251001"]
        assert calls["n"] == 3

    def test_does_not_retry_auth_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls = {"n": 0}

        def fake_urlopen(_req: object, timeout: float) -> object:
            calls["n"] += 1
            raise self._http_error(401)

        monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
        monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
        monkeypatch.setattr(runner_retry.time, "sleep", lambda _delay: None)

        assert claude_runner._anthropic_model_ids() is None
        assert calls["n"] == 1

    def test_gives_up_after_max_attempts(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls = {"n": 0}

        def fake_urlopen(_req: object, timeout: float) -> object:
            calls["n"] += 1
            raise TimeoutError

        monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
        monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
        monkeypatch.setattr(runner_retry.time, "sleep", lambda _delay: None)

        assert claude_runner._anthropic_model_ids() is None
        assert calls["n"] == claude_runner._MODELS_API_RETRY.max_attempts


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
        """Capture the `subprocess.run` call and return a clean fake proc."""

        def fake_run(cmd: list[str], **kw: Any) -> Any:
            captured["cmd"] = cmd
            captured["cwd"] = kw.get("cwd")
            captured["env"] = kw.get("env")
            return _proc()

        monkeypatch.setattr(subprocess, "run", fake_run)
        monkeypatch.setattr(
            claude_runner,
            "parse_stream_json",
            lambda _stdout, _skill, _root=None: (True, "answer", [], "resolved-model"),
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

    def test_runs_under_isolated_home(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setenv("HOME", "/real/home")
        captured: dict[str, Any] = {}
        self._stub_subprocess(monkeypatch, captured)

        ClaudeRunner().run("do it", tmp_path, "demo", model="claude-haiku-4-5")

        env = captured["env"]
        assert env["HOME"] != "/real/home"


class TestClaudeRunnerRunErrored:
    """Errored trials are retried, then marked rather than graded as fails."""

    @staticmethod
    def _stub(
        monkeypatch: pytest.MonkeyPatch, procs: list[Any]
    ) -> dict[str, int]:
        """Feed `procs` to successive subprocess calls; count the attempts."""
        calls = {"n": 0}

        def fake_run(cmd: list[str], **kw: Any) -> Any:
            proc = procs[min(calls["n"], len(procs) - 1)]
            calls["n"] += 1
            if isinstance(proc, BaseException):
                raise proc
            return proc

        monkeypatch.setattr(subprocess, "run", fake_run)
        monkeypatch.setattr(runner_retry.time, "sleep", lambda _delay: None)
        return calls

    def test_nonzero_exit_marks_run_errored_after_retries(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        calls = self._stub(
            monkeypatch, [_proc(stdout="", returncode=1, stderr="boom")]
        )

        run = ClaudeRunner().run("do it", tmp_path, "demo", model="haiku")

        assert run.errored is True
        assert run.skill_invoked is False
        assert "CLI exited with status 1" in run.error
        assert "boom" in run.error
        assert calls["n"] == TRIAL_RETRY.max_attempts

    def test_is_error_result_event_marks_run_errored(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        self._stub(monkeypatch, [_proc(stdout=_ERROR_STREAM)])

        run = ClaudeRunner().run("do it", tmp_path, "demo", model="haiku")

        assert run.errored is True
        assert "error_during_execution" in run.error
        assert "API Error: 500" in run.error

    def test_empty_stream_marks_run_errored(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        self._stub(monkeypatch, [_proc(stdout="")])

        run = ClaudeRunner().run("do it", tmp_path, "demo", model="haiku")

        assert run.errored is True
        assert "no assistant events" in run.error

    def test_transient_error_retries_then_succeeds(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        calls = self._stub(
            monkeypatch, [_proc(stdout=_ERROR_STREAM), _proc()]
        )

        run = ClaudeRunner().run("do it", tmp_path, "demo", model="haiku")

        assert run.errored is False
        assert run.assistant_text == "hi"
        assert calls["n"] == 2

    def test_timeout_marks_run_errored(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        self._stub(
            monkeypatch,
            [subprocess.TimeoutExpired(cmd="claude", timeout=300)],
        )

        run = ClaudeRunner().run(
            "do it", tmp_path, "demo", timeout=1, model="haiku"
        )

        assert run.errored is True
        assert "timed out" in run.error
