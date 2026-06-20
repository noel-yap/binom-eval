"""Unit tests for `binom_eval.stream_json` (stream-json parsing).

`binom_eval` is skill-independent, so its logic is tested once here rather
than duplicated in every per-skill `evals/test_helpers.py`. Those per-skill
files keep only the tests for their thin, SKILL_NAME-bound wrappers.
"""

from __future__ import annotations

import json

from binom_eval import (
    _assistant_content_blocks,
    _content_blocks_from_event,
    _is_assistant_event,
    _is_skill_hit,
    _message_from_event,
    _text_from_block,
    _try_parse_json,
    parse_stream_json,
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
        assert parse_stream_json("", SKILL) == (False, "", [])

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
        invoked, text, tools = parse_stream_json(
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
        invoked, _, _ = parse_stream_json(line, SKILL)
        assert invoked is False

    def test_ignores_unparseable_lines(self) -> None:
        line = json.dumps(
            {
                "type": "assistant",
                "message": {"content": [{"type": "text", "text": "ok"}]},
            }
        )
        invoked, text, _ = parse_stream_json("garbage line\n" + line, SKILL)
        assert (invoked, text) == (False, "ok")