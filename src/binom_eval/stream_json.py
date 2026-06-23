"""Parsing of `claude -p --output-format stream-json` output.

Turns the raw stdout of a single `claude -p` run into an `EvalRun`: the
`EvalRun` dataclass is the shared currency every downstream layer passes
around, and `parse_stream_json` is the one public entry point. The private
block helpers exist only to flatten assistant-event content into the three
facts an eval cares about: whether the skill fired, the assistant text, and
the tool-use blocks.
"""

from __future__ import annotations

import contextlib
import json
from dataclasses import dataclass, field
from typing import Any


@dataclass
class EvalRun:
    eval_id: str
    prompt: str
    skill_invoked: bool
    assistant_text: str
    tool_uses: list[dict[str, Any]] = field(default_factory=list)


def _try_parse_json(line: str) -> dict[str, Any] | None:
    """Parse one stream-json line into a dict, or None if it isn't valid JSON.

    Args:
        line: A single line of `claude -p` stdout.

    Returns:
        The decoded object when `line` is a JSON object, else None. Non-JSON
        lines (blank lines, partial chunks) are silently skipped.
    """
    result: dict[str, Any] | None = None
    with contextlib.suppress(json.JSONDecodeError, ValueError, AttributeError):
        result = json.loads(line.strip())
    return result


def _is_assistant_event(ev: dict[str, Any]) -> bool:
    """True if `ev` is an `assistant` stream-json event."""
    return ev.get("type") == "assistant"


def _message_from_event(ev: dict[str, Any]) -> dict[str, Any]:
    """The event's `message` payload, or an empty dict when absent."""
    msg = ev.get("message")
    return msg if msg is not None else {}


def _content_blocks_from_event(ev: dict[str, Any]) -> list[dict[str, Any]]:
    """The event message's content blocks, or an empty list when absent."""
    content = _message_from_event(ev).get("content")
    return content if content else []


def _assistant_content_blocks(
    events: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Flatten the content blocks of every assistant event in `events`."""
    assistant_events = filter(_is_assistant_event, events)
    return [
        b for ev in assistant_events for b in _content_blocks_from_event(ev)
    ]


def _is_skill_hit(block: dict[str, Any], skill_name: str) -> bool:
    """True if `block` is a Skill tool_use whose input names `skill_name`."""
    return all(
        [
            block.get("name") == "Skill",
            skill_name in str(block.get("input", {})),
        ]
    )


def _text_from_block(block: dict[str, Any]) -> str | None:
    """The text of a `text` block, or None for any other block type."""
    return block.get("text", "") if block.get("type") == "text" else None


def parse_stream_json(
    stdout: str, skill_name: str
) -> tuple[bool, str, list[dict[str, Any]]]:
    """Parse `claude -p --output-format stream-json` stdout into
    (skill_invoked, assistant_text, tool_uses)."""
    events = list(filter(None, map(_try_parse_json, stdout.splitlines())))
    blocks = _assistant_content_blocks(events)
    skill_invoked = any(_is_skill_hit(b, skill_name) for b in blocks)
    text = "\n".join(filter(None, map(_text_from_block, blocks)))
    tool_uses = list(filter(lambda b: b.get("type") == "tool_use", blocks))
    return skill_invoked, text, tool_uses


def tool_invoked(run: EvalRun, tool_name: str, target: str) -> bool:
    """True when ``tool_name`` was used with ``target`` in its input payload."""
    for block in run.tool_uses:
        if block.get("name") != tool_name:
            continue
        if target in str(block.get("input", {})):
            return True
    return False


def agent_invoked(run: EvalRun, agent_name: str) -> bool:
    """True when the Agent tool was used with ``agent_name``."""
    return tool_invoked(run, "Agent", agent_name)


def skill_invoked_in_tools(run: EvalRun, skill_name: str) -> bool:
    """True when the Skill tool was used with ``skill_name``."""
    return tool_invoked(run, "Skill", skill_name)