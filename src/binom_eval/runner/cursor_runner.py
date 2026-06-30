"""The `cursor-agent` backend: `CursorRunner`.

The concrete `Runner` for Cursor's headless CLI. Like `ClaudeRunner` it owns
every backend-specific detail -- the `cursor-agent` binary, the `--print`
invocation flags, the `--version` probe, and the model check via
`--list-models` -- so the rest of the harness keeps depending only on the
backend-agnostic `Runner` interface.

`cursor-agent --output-format stream-json` shares two shapes with Claude
Code's stream-json -- a `system`/`init` event carrying the model name and
`assistant` events whose `message.content` holds `{"type": "text"}` blocks --
so the text and model are parsed by reusing `parse_stream_json`. The tool
record differs: Cursor emits tool calls as separate top-level `tool_call`
events rather than `tool_use` blocks nested in the assistant message, so
`parse_cursor_stream_json` translates each started `tool_call` into the
harness's `{"type": "tool_use", "name", "input"}` convention. That lets the
existing `EvalRun` predicates (`tool_invoked`, `agent_invoked`,
`skill_invoked_in_tools`) work unchanged against a Cursor run.

Cursor loads skills by reading `{skill_name}/SKILL.md` via `readToolCall`
rather than emitting Claude Code's `Skill` tool, so skill invocation is also
detected when a translated Read targets that path. Detection is scoped to the
run's workspace: only a SKILL.md read under `repo_root` counts, so a trigger
pass means the project's own skill was used rather than a copy from a user
skill root (`~/.claude/skills`, `~/.cursor/skills`, ...) that may differ. The
run is also pinned to that workspace via `--workspace repo_root` so the
project's `.cursor/skills/`, `skills/`, and `.claude/skills/` trees are the
ones exposed to the agent.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Any

from binom_eval.runner import (
    DEFAULT_TIMEOUT_SECONDS,
    Runner,
    isolated_workdir,
    stripped_env,
)
from binom_eval.stream_json import (
    EvalRun,
    _is_skill_hit,
    _try_parse_json,
    parse_stream_json,
)


def _is_started_tool_call(event: dict[str, Any]) -> bool:
    """True if `event` is a `tool_call` `started` event."""
    return event.get("type") == "tool_call" and event.get("subtype") == (
        "started"
    )


def _cursor_tool_use(event: dict[str, Any]) -> dict[str, Any] | None:
    """Translate one Cursor `tool_call` event into a `tool_use` block.

    Cursor names a tool by the single key under `tool_call`: either a typed
    `<verb>ToolCall` (e.g. `readToolCall`, `writeToolCall`) carrying an `args`
    payload, or a generic `function` carrying `{"name", "arguments"}`. Both are
    flattened to the harness's `{"type": "tool_use", "name", "input"}` shape so
    the shared `EvalRun` predicates can match on tool name and target. Returns
    None when the event carries no recognizable tool payload.
    """
    call = event.get("tool_call")
    if not isinstance(call, dict):
        return None
    for key, payload in call.items():
        payload = payload if isinstance(payload, dict) else {}
        if key == "function":
            return {
                "type": "tool_use",
                "name": payload.get("name", ""),
                "input": payload.get("arguments", ""),
            }
        if key.endswith("ToolCall"):
            base = key[: -len("ToolCall")]
            return {
                "type": "tool_use",
                "name": base[:1].upper() + base[1:],
                "input": payload.get("args", {}),
            }
    return None


def _cursor_skill_read_hit(
    block: dict[str, Any], skill_name: str, repo_root: Path
) -> bool:
    """True when Cursor read this project's skill SKILL.md via `readToolCall`.

    Cursor discovers skills at startup and pulls instructions on demand by
    reading `{skill_name}/SKILL.md`. A skill may be present both in the project
    (`{repo_root}/.cursor/skills/`, `{repo_root}/skills/`,
    `{repo_root}/.claude/skills/`) and in a user skill root
    (`~/.claude/skills`, `~/.cursor/skills`, ...). Only a read whose path names
    `{skill_name}/SKILL.md` *and* resolves under `repo_root` counts as a hit,
    so a trigger pass attests to the project's own skill rather than a global
    copy that may differ.
    """
    if block.get("name") != "Read":
        return False
    raw = str(block.get("input", {}).get("path", "")).replace(chr(92), "/")
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


def parse_cursor_stream_json(
    stdout: str, skill_name: str, repo_root: Path
) -> tuple[bool, str, list[dict[str, Any]], str]:
    """Parse `cursor-agent --output-format stream-json` stdout into
    (skill_invoked, assistant_text, tool_uses, model).

    Assistant text and the model reuse `parse_stream_json` (Cursor and Claude
    share those event shapes); the tool record is rebuilt from Cursor's
    top-level `tool_call` events, and the skill verdict is taken from either
    parser's view of the translated tool calls. `repo_root` scopes the
    SKILL.md read check to the run's workspace so a trigger pass attests to the
    project's own skill rather than a user skill root.
    """
    skill_invoked, text, _claude_tool_uses, model = parse_stream_json(
        stdout, skill_name
    )
    events = filter(None, map(_try_parse_json, stdout.splitlines()))
    tool_uses = list(
        filter(
            None,
            (
                _cursor_tool_use(ev)
                for ev in events
                if _is_started_tool_call(ev)
            ),
        )
    )
    skill_invoked = skill_invoked or any(
        _is_skill_hit(block, skill_name)
        or _cursor_skill_read_hit(block, skill_name, repo_root)
        for block in tool_uses
    )
    return skill_invoked, text, tool_uses, model


class CursorRunner(Runner):
    """A `Runner` backed by the `cursor-agent` CLI."""

    def version(self) -> str:
        """Return the `cursor-agent` version string, or '' when not on PATH."""
        try:
            result = subprocess.run(
                ["cursor-agent", "--version"],
                capture_output=True,
                text=True,
                timeout=10,
                env=stripped_env(),
            )
            return result.stdout.strip()
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return ""

    def preflight(self) -> str | None:
        """Return why `cursor-agent` cannot run, or None when ready.

        Requires the `cursor-agent` CLI on PATH. Unlike Claude, Cursor
        authenticates from its own stored session (`cursor-agent login`)
        rather than a single environment variable, so there is no credential
        env var to check here; an unauthenticated CLI surfaces on the first
        trial.
        """
        if shutil.which("cursor-agent") is None:
            return "cursor-agent CLI not found on PATH"
        return None

    def validate_model(self, model: str, timeout: int = 30) -> str | None:
        """Confirm `cursor-agent` can use `model`; return an error or None.

        Checks `model` against `cursor-agent --list-models` rather than
        spawning a probe run. Returns None when the model is listed -- and,
        defensively, whenever the list cannot be read (CLI absent, timeout, or
        empty output) so a flaky check never blocks a run that might otherwise
        succeed. Only a positively-read list that omits `model` yields an error.
        """
        try:
            proc = subprocess.run(
                ["cursor-agent", "--list-models"],
                capture_output=True,
                text=True,
                env=stripped_env(),
                timeout=timeout,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return None
        listed = proc.stdout.split()
        if not listed or model in listed:
            return None
        return f"model not found: {model}"

    def run(
        self,
        prompt: str,
        repo_root: Path,
        skill_name: str,
        timeout: int = DEFAULT_TIMEOUT_SECONDS,
        *,
        isolate: bool = False,
        model: str,
    ) -> EvalRun:
        """Invoke `cursor-agent --print` once and parse its stream-json output.

        With `isolate=True` the call runs in a throwaway copy of `repo_root`
        (see `isolated_workdir`) so a skill that writes to the tree cannot
        affect `repo_root` or a concurrent run; otherwise it runs in
        `repo_root` directly. `--force` and `--trust` keep the headless run
        from blocking on command or workspace-trust prompts. The run is pinned
        to the working tree with `--workspace` (and a matching `cwd`) so the
        project's own `.cursor/skills/`, `skills/`, and `.claude/skills/` trees
        are the ones exposed to the agent, and skill detection is scoped to
        that same tree. `model` is assumed to be set and is always forwarded as
        `--model`.
        """
        with isolated_workdir(repo_root, isolate) as workdir:
            cmd = [
                "cursor-agent",
                "--print",
                "--output-format",
                "stream-json",
                "--force",
                "--trust",
                "--workspace",
                str(workdir),
                "--model",
                model,
                prompt,
            ]
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=str(workdir),
                env=stripped_env(),
                timeout=timeout,
            )
            skill_invoked, assistant_text, tool_uses, model = (
                parse_cursor_stream_json(proc.stdout, skill_name, workdir)
            )
        return EvalRun(
            eval_id="",
            prompt=prompt,
            skill_invoked=skill_invoked,
            assistant_text=assistant_text,
            tool_uses=tool_uses,
            model=model,
        )
