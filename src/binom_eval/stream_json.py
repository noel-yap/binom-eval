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
from pathlib import Path
from typing import Any


@dataclass
class EvalRun:
    eval_id: str
    prompt: str
    skill_invoked: bool
    assistant_text: str
    tool_uses: list[dict[str, Any]] = field(default_factory=list)
    model: str = ""
    prompt_input: str = ""
    # A trial that could not produce a gradable transcript (CLI died, API
    # error surfaced as an `is_error` result, retries exhausted). Errored
    # trials are excluded from the Beta-binomial counts -- an infrastructure
    # failure says nothing about the skill's true pass rate -- rather than
    # graded as behavioral failures. `error` carries the last failure reason.
    errored: bool = False
    error: str = ""


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


def _skill_read_hit(
    block: dict[str, Any], skill_name: str, repo_root: Path
) -> bool:
    """True when a Read tool_use opened this project's `{skill_name}/SKILL.md`.

    Agents may invoke a skill via the Skill tool or by reading its SKILL.md
    directly. Only a read whose path names `{skill_name}/SKILL.md` and resolves
    under `repo_root` counts, so a trigger pass attests to the project's own
    skill rather than a copy from a user skill root that may differ.
    """
    if block.get("name") != "Read":
        return False
    payload = block.get("input", {})
    if not isinstance(payload, dict):
        return False
    raw = str(payload.get("path") or payload.get("file_path") or "").replace(
        chr(92), "/"
    )
    if not raw or f"/{skill_name}/SKILL.md" not in raw:
        return False
    root = repo_root.resolve()
    candidate = Path(raw)
    if not candidate.is_absolute():
        candidate = root / candidate
    try:
        resolved = candidate.resolve()
    except OSError:
        return False
    return resolved == root or root in resolved.parents


def _text_from_block(block: dict[str, Any]) -> str | None:
    """The text of a `text` block, or None for any other block type."""
    return block.get("text", "") if block.get("type") == "text" else None


def _model_from_events(events: list[dict[str, Any]]) -> str:
    """Extract the model name from the system init event, or '' when absent."""
    for ev in events:
        if ev.get("type") == "system" and ev.get("subtype") == "init":
            model = ev.get("model", "")
            if model:
                return str(model)
    return ""


def stream_error(stdout: str) -> str | None:
    """Why a run's stream-json output signals an errored trial, or None.

    A trial is errored -- as opposed to a behavioral failure -- when the
    stream carries a `result` event with `is_error: true` (API errors such as
    500/overload surface this way, with subtypes like `error_during_execution`
    or `error_max_turns`), or when it contains no assistant events at all
    (the CLI died before the model produced anything). Kept pure so the
    detection is unit-testable without spawning a CLI.
    """
    events = list(filter(None, map(_try_parse_json, stdout.splitlines())))
    for ev in events:
        if ev.get("type") == "result" and ev.get("is_error"):
            subtype = str(ev.get("subtype") or "error")
            message = str(ev.get("result") or "").strip()
            return f"{subtype}: {message}" if message else subtype
    if not any(map(_is_assistant_event, events)):
        return "no assistant events in stream-json output"
    return None


def parse_stream_json(
    stdout: str,
    skill_name: str,
    repo_root: Path | None = None,
) -> tuple[bool, str, list[dict[str, Any]], str]:
    """Parse `claude -p --output-format stream-json` stdout into
    (skill_invoked, assistant_text, tool_uses, model).

    When `repo_root` is given, a Read of `{skill_name}/SKILL.md` under that
    tree also counts as a skill invocation (some backends read the file rather
    than emitting a Skill tool_use).
    """
    events = list(filter(None, map(_try_parse_json, stdout.splitlines())))
    blocks = _assistant_content_blocks(events)
    tool_uses = list(filter(lambda b: b.get("type") == "tool_use", blocks))
    skill_invoked = any(_is_skill_hit(b, skill_name) for b in blocks)
    if repo_root is not None:
        skill_invoked = skill_invoked or any(
            _skill_read_hit(block, skill_name, repo_root) for block in tool_uses
        )
    text = "\n".join(filter(None, map(_text_from_block, blocks)))
    model = _model_from_events(events)
    return skill_invoked, text, tool_uses, model


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


def skill_was_invoked(run: EvalRun, skill_name: str) -> bool:
    """True when the skill fired via stream flag or Skill tool."""
    return run.skill_invoked or skill_invoked_in_tools(run, skill_name)


def agent_or_skill_invoked(
    run: EvalRun, agent_name: str, skill_name: str
) -> bool:
    """True when ``agent_name`` or ``skill_name`` was invoked via tools."""
    return agent_invoked(run, agent_name) or skill_invoked_in_tools(
        run, skill_name
    )