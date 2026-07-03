"""Unit tests for `binom_eval.stream_json` (stream-json parsing).

`binom_eval` is skill-independent, so its logic is tested once here rather
than duplicated in every per-skill `evals/test_helpers.py`. Those per-skill
files keep only the tests for their thin, SKILL_NAME-bound wrappers.
"""

from __future__ import annotations

import json
from pathlib import Path

from binom_eval import (
    _assistant_content_blocks,
    _content_blocks_from_event,
    _is_assistant_event,
    _is_skill_hit,
    _message_from_event,
    _skill_read_hit,
    _text_from_block,
    _try_parse_json,
    agent_invoked,
    parse_stream_json,
    skill_invoked_in_tools,
    skill_was_invoked,
    agent_or_skill_invoked,
    stream_error,
    tool_invoked,
)

SKILL = "demo-skill"


class TestTryParseJson:
    def test_valid_returns_dict(self) -> None:
        assert _try_parse_json('{"type": "assistant"}') == {"type": "assistant"}

    def test_invalid_returns_none(self) -> None:
        assert _try_parse_json("not json") is None

    def test_empty_line_returns_none(self) -> None:
        assert _try_parse_json("") is None

    def test_whitespace_stripped(self) -> None:
        assert _try_parse_json('  {"a": 1}  ') == {"a": 1}


class TestIsAssistantEvent:
    def test_true_when_type_assistant(self) -> None:
        assert _is_assistant_event({"type": "assistant"}) is True

    def test_false_when_type_other(self) -> None:
        assert _is_assistant_event({"type": "user"}) is False

    def test_false_when_type_absent(self) -> None:
        assert _is_assistant_event({}) is False


class TestMessageFromEvent:
    def test_returns_message_when_present(self) -> None:
        assert _message_from_event({"message": {"a": 1}}) == {"a": 1}

    def test_returns_empty_dict_when_msg_none(self) -> None:
        assert _message_from_event({"message": None}) == {}

    def test_returns_empty_dict_when_msg_absent(self) -> None:
        assert _message_from_event({}) == {}


class TestContentBlocksFromEvent:
    def test_returns_content_list(self) -> None:
        ev = {"message": {"content": [{"type": "text", "text": "hi"}]}}
        assert _content_blocks_from_event(ev) == [
            {"type": "text", "text": "hi"}
        ]

    def test_returns_empty_list_when_content_missing(self) -> None:
        assert _content_blocks_from_event({"message": {}}) == []

    def test_returns_empty_list_when_content_none(self) -> None:
        assert _content_blocks_from_event({"message": {"content": None}}) == []

    def test_returns_empty_list_when_message_missing(self) -> None:
        assert _content_blocks_from_event({}) == []


class TestAssistantContentBlocks:
    def test_filters_to_assistant_only(self) -> None:
        events = [
            {
                "type": "user",
                "message": {"content": [{"type": "text", "text": "u"}]},
            },
            {
                "type": "assistant",
                "message": {"content": [{"type": "text", "text": "a"}]},
            },
        ]
        assert _assistant_content_blocks(events) == [
            {"type": "text", "text": "a"}
        ]

    def test_empty_events_returns_empty(self) -> None:
        assert _assistant_content_blocks([]) == []


class TestIsSkillHit:
    def test_matches_skill_name_in_input(self) -> None:
        block = {"name": "Skill", "input": {"skill": SKILL}}
        assert _is_skill_hit(block, SKILL) is True

    def test_rejects_non_skill_tool(self) -> None:
        block = {"name": "Read", "input": {"skill": SKILL}}
        assert _is_skill_hit(block, SKILL) is False

    def test_rejects_wrong_skill_name(self) -> None:
        block = {"name": "Skill", "input": {"skill": "other-skill"}}
        assert _is_skill_hit(block, SKILL) is False


class TestTextFromBlock:
    def test_returns_text_for_text_block(self) -> None:
        assert _text_from_block({"type": "text", "text": "hello"}) == "hello"

    def test_returns_none_when_type_not_text(self) -> None:
        assert _text_from_block({"type": "tool_use", "text": "x"}) is None

    def test_returns_empty_string_when_text_key_absent(self) -> None:
        assert _text_from_block({"type": "text"}) == ""


class TestParseStreamJson:
    def test_no_skill_no_text(self) -> None:
        assert parse_stream_json("", SKILL) == (False, "", [], "")

    def test_detects_skill_invocation_and_text(self) -> None:
        events = [
            {
                "type": "assistant",
                "message": {
                    "content": [
                        {
                            "type": "tool_use",
                            "name": "Skill",
                            "input": {"skill": SKILL},
                        },
                        {"type": "text", "text": "did the thing"},
                    ]
                },
            }
        ]
        invoked, text, tools, model = parse_stream_json(
            "\n".join(map(json.dumps, events)), SKILL
        )
        assert invoked is True
        assert text == "did the thing"
        assert tools[0]["name"] == "Skill"

    def test_no_skill_returns_false(self) -> None:
        line = json.dumps(
            {
                "type": "assistant",
                "message": {
                    "content": [{"type": "text", "text": "plain reply"}]
                },
            }
        )
        invoked, *_ = parse_stream_json(line, SKILL)
        assert invoked is False

    def test_ignores_unparseable_lines(self) -> None:
        line = json.dumps(
            {
                "type": "assistant",
                "message": {"content": [{"type": "text", "text": "ok"}]},
            }
        )
        invoked, text, *_ = parse_stream_json("garbage line\n" + line, SKILL)
        assert (invoked, text) == (False, "ok")

    def test_counts_skill_read_under_repo_root(self, tmp_path: Path) -> None:
        skill_dir = tmp_path / "skills" / SKILL
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text("skill body", encoding="utf-8")
        events = [
            {
                "type": "assistant",
                "message": {
                    "content": [
                        {
                            "type": "tool_use",
                            "name": "Read",
                            "input": {
                                "file_path": str(skill_dir / "SKILL.md"),
                            },
                        }
                    ]
                },
            }
        ]
        invoked, _, tools, _ = parse_stream_json(
            "\n".join(map(json.dumps, events)), SKILL, tmp_path
        )
        assert invoked is True
        assert tools[0]["name"] == "Read"


class TestSkillReadHit:
    def test_matches_project_skill_read(self, tmp_path: Path) -> None:
        skill_dir = tmp_path / ".claude" / "skills" / "demo"
        skill_dir.mkdir(parents=True)
        skill_file = skill_dir / "SKILL.md"
        skill_file.write_text("x", encoding="utf-8")
        block = {
            "name": "Read",
            "input": {"path": str(skill_file)},
        }
        assert _skill_read_hit(block, "demo", tmp_path)

    def test_rejects_user_skill_root_outside_repo(self, tmp_path: Path) -> None:
        block = {
            "name": "Read",
            "input": {"path": "/Users/me/.claude/skills/demo/SKILL.md"},
        }
        assert not _skill_read_hit(block, "demo", tmp_path)



class TestModelFromEvents:
    def test_extracts_model_from_system_init(self) -> None:
        from binom_eval import _model_from_events

        events = [
            {"type": "system", "subtype": "init", "model": "claude-sonnet-4-6"}
        ]
        assert _model_from_events(events) == "claude-sonnet-4-6"

    def test_returns_empty_when_no_system_event(self) -> None:
        from binom_eval import _model_from_events

        assert _model_from_events([]) == ""
        assert _model_from_events([{"type": "assistant"}]) == ""

    def test_returns_empty_when_model_key_absent(self) -> None:
        from binom_eval import _model_from_events

        events = [{"type": "system", "subtype": "init"}]
        assert _model_from_events(events) == ""

    def test_ignores_non_init_system_events(self) -> None:
        from binom_eval import _model_from_events

        events = [{"type": "system", "subtype": "other", "model": "x"}]
        assert _model_from_events(events) == ""

    def test_parse_stream_json_extracts_model(self) -> None:
        import json as _json

        events = [
            {"type": "system", "subtype": "init", "model": "claude-haiku-4-5"},
            {
                "type": "assistant",
                "message": {
                    "content": [{"type": "text", "text": "hi"}]
                },
            },
        ]
        _, _, _, model = parse_stream_json(
            "\n".join(map(_json.dumps, events)), SKILL
        )
        assert model == "claude-haiku-4-5"

class TestToolInvocation:
    def _run(self, tool_name: str, tool_input: dict) -> EvalRun:
        from binom_eval import EvalRun

        return EvalRun(
            eval_id="t",
            prompt="",
            skill_invoked=False,
            assistant_text="",
            tool_uses=[
                {"type": "tool_use", "name": tool_name, "input": tool_input},
            ],
        )

    def test_agent_invoked(self) -> None:
        run = self._run("Agent", {"name": "dependency-injection"})
        assert agent_invoked(run, "dependency-injection") is True
        assert agent_invoked(run, "other") is False

    def test_skill_invoked_in_tools(self) -> None:
        run = self._run("Skill", {"skill": SKILL})
        assert skill_invoked_in_tools(run, SKILL) is True
        assert skill_invoked_in_tools(run, "other-skill") is False

    def test_tool_invoked_rejects_other_tool(self) -> None:
        run = self._run("Read", {"skill": SKILL})
        assert tool_invoked(run, "Skill", SKILL) is False

    def test_tool_invoked_returns_false_when_tool_uses_empty(self) -> None:
        from binom_eval import EvalRun

        run = EvalRun(
            eval_id="t",
            prompt="",
            skill_invoked=False,
            assistant_text="",
            tool_uses=[],
        )
        assert tool_invoked(run, "Skill", SKILL) is False

    def test_tool_invoked_returns_true_when_target_appears_in_input(self) -> None:
        run = self._run("Skill", {"skill": SKILL})
        assert tool_invoked(run, "Skill", SKILL) is True

    def test_tool_invoked_returns_false_when_tool_matches_but_target_missing(
        self,
    ) -> None:
        run = self._run("Skill", {"skill": "other-skill"})
        assert tool_invoked(run, "Skill", SKILL) is False

    def test_tool_invoked_finds_match_in_later_tool_use(self) -> None:
        from binom_eval import EvalRun

        run = EvalRun(
            eval_id="t",
            prompt="",
            skill_invoked=False,
            assistant_text="",
            tool_uses=[
                {"type": "tool_use", "name": "Read", "input": {"path": "/tmp"}},
                {"type": "tool_use", "name": "Skill", "input": {"skill": SKILL}},
            ],
        )
        assert tool_invoked(run, "Skill", SKILL) is True

    def test_skill_was_invoked_from_stream_flag(self) -> None:
        from binom_eval import EvalRun

        run = EvalRun(
            eval_id="t",
            prompt="",
            skill_invoked=True,
            assistant_text="",
            tool_uses=[],
        )
        assert skill_was_invoked(run, SKILL) is True

    def test_agent_or_skill_invoked(self) -> None:
        run = self._run("Agent", {"name": "dependency-injection"})
        assert agent_or_skill_invoked(
            run, "dependency-injection", "dependency-injection"
        )
        assert not agent_or_skill_invoked(run, "other", "other-skill")

class TestStreamError:
    """Detection of errored trials from stream-json output."""

    _ASSISTANT = (
        '{"type":"assistant","message":{"content":'
        '[{"type":"text","text":"hi"}]}}'
    )
    _OK_RESULT = (
        '{"type":"result","subtype":"success","is_error":false,"result":"ok"}'
    )

    def test_none_for_clean_stream(self) -> None:
        assert stream_error(f"{self._ASSISTANT}\n{self._OK_RESULT}\n") is None

    def test_reports_is_error_result_with_subtype_and_message(self) -> None:
        stdout = (
            f"{self._ASSISTANT}\n"
            '{"type":"result","subtype":"error_during_execution",'
            '"is_error":true,"result":"API Error: 500"}\n'
        )
        error = stream_error(stdout)
        assert error == "error_during_execution: API Error: 500"

    def test_reports_is_error_result_without_message(self) -> None:
        stdout = (
            f"{self._ASSISTANT}\n"
            '{"type":"result","subtype":"error_max_turns","is_error":true}\n'
        )
        assert stream_error(stdout) == "error_max_turns"

    def test_reports_stream_with_no_assistant_events(self) -> None:
        stdout = f"{self._OK_RESULT}\n"
        error = stream_error(stdout)
        assert error is not None and "no assistant events" in error

    def test_reports_empty_stdout(self) -> None:
        error = stream_error("")
        assert error is not None and "no assistant events" in error

    def test_non_error_result_does_not_mask_missing_assistant_events(
        self,
    ) -> None:
        # A dead CLI can still flush non-assistant events; those alone are
        # not a gradeable transcript.
        stdout = '{"type":"system","subtype":"init","model":"m"}\n'
        assert stream_error(stdout) is not None
