"""Unit tests for `binom_eval.runner.cursor_runner`.

`CursorRunner.run` spawns a real `cursor-agent` subprocess, so its full path
is exercised by the live evals; here the subprocess and the stream-json parser
are stubbed so the command-assembly and result-wiring branches can be checked
without invoking the CLI. `version` and `validate_model` stub only
`subprocess.run`. The pure parsing helpers (`_is_started_tool_call`,
`_cursor_tool_use`, `parse_cursor_stream_json`) are tested directly.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

import pytest

from binom_eval import EvalRun
from binom_eval.runner import cursor_runner
from binom_eval.runner.cursor_runner import (
    CursorRunner,
    _cursor_tool_use,
    _is_started_tool_call,
    parse_cursor_stream_json,
)


class TestIsStartedToolCall:
    def test_true_for_started_tool_call(self) -> None:
        assert _is_started_tool_call(
            {"type": "tool_call", "subtype": "started"}
        )

    def test_false_when_subtype_is_not_started(self) -> None:
        assert not _is_started_tool_call(
            {"type": "tool_call", "subtype": "completed"}
        )

    def test_false_when_type_is_not_tool_call(self) -> None:
        assert not _is_started_tool_call(
            {"type": "assistant", "subtype": "started"}
        )


class TestCursorToolUse:
    def test_returns_none_when_tool_call_missing(self) -> None:
        assert _cursor_tool_use({"type": "tool_call"}) is None

    def test_returns_none_when_tool_call_not_a_dict(self) -> None:
        assert _cursor_tool_use({"tool_call": "nope"}) is None

    def test_returns_none_for_unrecognized_payload_key(self) -> None:
        assert _cursor_tool_use({"tool_call": {"mysteryThing": {}}}) is None

    def test_maps_typed_tool_call_to_capitalized_name(self) -> None:
        block = _cursor_tool_use(
            {"tool_call": {"readToolCall": {"args": {"path": "a.txt"}}}}
        )
        assert block == {
            "type": "tool_use",
            "name": "Read",
            "input": {"path": "a.txt"},
        }

    def test_maps_write_tool_call(self) -> None:
        block = _cursor_tool_use(
            {"tool_call": {"writeToolCall": {"args": {"path": "b.txt"}}}}
        )
        assert block is not None
        assert block["name"] == "Write"
        assert block["input"] == {"path": "b.txt"}

    def test_typed_payload_defaults_to_empty_input_when_not_a_dict(
        self,
    ) -> None:
        block = _cursor_tool_use({"tool_call": {"readToolCall": None}})
        assert block == {"type": "tool_use", "name": "Read", "input": {}}

    def test_maps_function_call_to_name_and_arguments(self) -> None:
        block = _cursor_tool_use(
            {"tool_call": {"function": {"name": "Grep", "arguments": "foo"}}}
        )
        assert block == {
            "type": "tool_use",
            "name": "Grep",
            "input": "foo",
        }


def _event_lines(*objects: str) -> str:
    return "\n".join(objects) + "\n"


_SYSTEM_INIT = (
    '{"type":"system","subtype":"init","model":"GPT-5","session_id":"s"}'
)
_ASSISTANT_HI = (
    '{"type":"assistant","message":{"role":"assistant",'
    '"content":[{"type":"text","text":"hello"}]}}'
)
_ASSISTANT_BYE = (
    '{"type":"assistant","message":{"role":"assistant",'
    '"content":[{"type":"text","text":"bye"}]}}'
)
_READ_STARTED = (
    '{"type":"tool_call","subtype":"started","call_id":"c1",'
    '"tool_call":{"readToolCall":{"args":{"path":"README.md"}}}}'
)
_READ_COMPLETED = (
    '{"type":"tool_call","subtype":"completed","call_id":"c1",'
    '"tool_call":{"readToolCall":{"args":{"path":"README.md"}}}}'
)


class TestParseCursorStreamJson:
    def test_aggregates_text_and_model_and_translates_tools(self) -> None:
        stdout = _event_lines(
            _SYSTEM_INIT, _ASSISTANT_HI, _READ_STARTED, _ASSISTANT_BYE
        )
        skill, text, tools, model = parse_cursor_stream_json(stdout, "demo")
        assert text == "hello\nbye"
        assert model == "GPT-5"
        assert tools == [
            {"type": "tool_use", "name": "Read", "input": {"path": "README.md"}}
        ]
        assert skill is False

    def test_counts_only_started_tool_calls(self) -> None:
        stdout = _event_lines(_READ_STARTED, _READ_COMPLETED)
        _skill, _text, tools, _model = parse_cursor_stream_json(stdout, "demo")
        assert len(tools) == 1

    def test_skill_detected_via_translated_cursor_tool_call(self) -> None:
        skill_call = (
            '{"type":"tool_call","subtype":"started",'
            '"tool_call":{"function":{"name":"Skill","arguments":"demo"}}}'
        )
        stdout = _event_lines(_ASSISTANT_HI, skill_call)
        skill, _text, _tools, _model = parse_cursor_stream_json(stdout, "demo")
        assert skill is True

    def test_skill_detected_via_claude_style_assistant_block(self) -> None:
        # parse_stream_json sees a nested Skill tool_use block while there are
        # no Cursor tool_call events, so the verdict comes from its view alone.
        assistant_skill = (
            '{"type":"assistant","message":{"role":"assistant","content":'
            '[{"type":"tool_use","name":"Skill","input":{"skill":"demo"}}]}}'
        )
        stdout = _event_lines(assistant_skill)
        skill, _text, _tools, _model = parse_cursor_stream_json(stdout, "demo")
        assert skill is True

    def test_skill_not_detected_for_unrelated_run(self) -> None:
        stdout = _event_lines(_ASSISTANT_HI, _READ_STARTED)
        skill, _text, _tools, _model = parse_cursor_stream_json(stdout, "demo")
        assert skill is False


class TestCursorRunnerVersion:
    def test_returns_stripped_version_on_success(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            subprocess,
            "run",
            lambda *a, **kw: type("R", (), {"stdout": "2025.01.0\n"})(),
        )
        assert CursorRunner().version() == "2025.01.0"

    def test_returns_empty_when_cli_absent(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def raise_fnf(*a: object, **kw: object) -> None:
            raise FileNotFoundError

        monkeypatch.setattr(subprocess, "run", raise_fnf)
        assert CursorRunner().version() == ""

    def test_returns_empty_on_timeout(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def raise_timeout(*a: object, **kw: object) -> None:
            raise subprocess.TimeoutExpired(cmd="cursor-agent", timeout=10)

        monkeypatch.setattr(subprocess, "run", raise_timeout)
        assert CursorRunner().version() == ""


class TestCursorRunnerPreflight:
    def test_ready_when_cli_present(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            cursor_runner.shutil, "which", lambda _n: "/usr/bin/cursor-agent"
        )
        assert CursorRunner().preflight() is None

    def test_reports_missing_cli(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(cursor_runner.shutil, "which", lambda _n: None)
        msg = CursorRunner().preflight()
        assert msg is not None and "cursor-agent CLI not found" in msg


class TestCursorRunnerValidateModel:
    @staticmethod
    def _stub_list_models(
        monkeypatch: pytest.MonkeyPatch, stdout: str
    ) -> None:
        monkeypatch.setattr(
            subprocess,
            "run",
            lambda *a, **kw: type("R", (), {"stdout": stdout})(),
        )

    def test_returns_none_when_cli_absent(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def raise_fnf(*a: object, **kw: object) -> None:
            raise FileNotFoundError

        monkeypatch.setattr(subprocess, "run", raise_fnf)
        assert CursorRunner().validate_model("sonnet-4.5") is None

    def test_returns_none_on_timeout(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def raise_timeout(*a: object, **kw: object) -> None:
            raise subprocess.TimeoutExpired(cmd="cursor-agent", timeout=1)

        monkeypatch.setattr(subprocess, "run", raise_timeout)
        assert CursorRunner().validate_model("sonnet-4.5") is None

    def test_returns_none_when_list_is_empty(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._stub_list_models(monkeypatch, "   \n")
        assert CursorRunner().validate_model("sonnet-4.5") is None

    def test_returns_none_when_model_is_listed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._stub_list_models(monkeypatch, "gpt-5\nsonnet-4.5\nopus-4.1\n")
        assert CursorRunner().validate_model("sonnet-4.5") is None

    def test_reports_error_when_model_absent_from_nonempty_list(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._stub_list_models(monkeypatch, "gpt-5\nopus-4.1\n")
        msg = CursorRunner().validate_model("sonnet-4.5")
        assert msg is not None and "sonnet-4.5" in msg


class TestCursorRunnerRun:
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
            cursor_runner,
            "parse_cursor_stream_json",
            lambda _stdout, _skill: (True, "answer", [], "resolved-model"),
        )

    def test_includes_model_flag_when_model_given(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        captured: dict[str, Any] = {}
        self._stub_subprocess(monkeypatch, captured)

        run = CursorRunner().run("do it", tmp_path, "demo", model="gpt-5")

        cmd = captured["cmd"]
        assert "--model" in cmd
        assert cmd[cmd.index("--model") + 1] == "gpt-5"
        assert cmd[-1] == "do it"
        assert isinstance(run, EvalRun)
        assert run.skill_invoked is True
        assert run.assistant_text == "answer"
        assert run.model == "resolved-model"

    def test_always_includes_model_flag(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        captured: dict[str, Any] = {}
        self._stub_subprocess(monkeypatch, captured)

        CursorRunner().run("do it", tmp_path, "demo", model="gpt-5")

        assert "--model" in captured["cmd"]
        assert captured["cmd"][-1] == "do it"

    def test_runs_in_repo_root_without_isolation(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        captured: dict[str, Any] = {}
        self._stub_subprocess(monkeypatch, captured)

        CursorRunner().run(
            "do it", tmp_path, "demo", isolate=False, model="gpt-5"
        )

        assert captured["cwd"] == str(tmp_path)
