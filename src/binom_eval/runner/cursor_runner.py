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

import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

from binom_eval.runner import (
    DEFAULT_TIMEOUT_SECONDS,
    Runner,
    _format_model_error,
    _run_trial,
    _spawn_checked,
    fake_home_env,
    isolated_workdir,
    stripped_env,
)
from binom_eval.stream_json import (
    EvalRun,
    _is_skill_hit,
    _skill_read_hit,
    _try_parse_json,
    parse_stream_json,
)

# On macOS the `cursor-agent` CLI persists its login/session under the
# `cursor-user` account in the user's *login* keychain. Evals run the CLI
# under a throwaway `HOME` (`fake_home_env`) with no login keychain
# present, so the CLI's keychain access fails -- macOS raises the blocking
# 'A keychain cannot be found to store "cursor-user."' popup that stalls
# headless runs. `AGENT_CLI_CREDENTIAL_STORE=memory` tells the CLI to keep
# credentials in-process instead of the macOS login keychain; the run then
# authenticates from `CURSOR_API_KEY`, which the harness always supplies.
# `CI=true` is a defensive non-interactive signal so the CLI never falls back
# to a login-keychain lock check or prompt.
KEYCHAIN_SKIP_ENV = {
    "AGENT_CLI_CREDENTIAL_STORE": "memory",
    "CI": "true",
}


def cursor_env(base: dict[str, str]) -> dict[str, str]:
    """Return `base` with the keychain-skip markers forced on.

    Layered over any base env (`stripped_env` for probes, `fake_home_env`
    for live runs) so every `cursor-agent` invocation bypasses the macOS
    login keychain and never raises the blocking 'cursor-user' popup.
    """
    return {**base, **KEYCHAIN_SKIP_ENV}


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
        or _skill_read_hit(block, skill_name, repo_root)
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
                env=cursor_env(stripped_env()),
            )
            return result.stdout.strip()
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return ""

    def preflight(self) -> str | None:
        """Return why `cursor-agent` cannot run, or None when ready.

        Requires the `cursor-agent` CLI on PATH and `CURSOR_API_KEY` set. Live
        runs execute under a throwaway `HOME` (`fake_home_env`) so the user's
        stored Cursor session is deliberately hidden -- evals must not pick up
        the invoking user's settings -- which leaves the API key as the only
        credential the headless run can authenticate with.
        """
        if shutil.which("cursor-agent") is None:
            return "cursor-agent CLI not found on PATH"
        if not os.environ.get("CURSOR_API_KEY"):
            return (
                "CURSOR_API_KEY is not set; live evals run under an isolated "
                "HOME (no stored login) and authenticate only via that key."
            )
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
                env=cursor_env(stripped_env()),
                timeout=timeout,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return None
        listed = proc.stdout.split()
        if not listed or model in listed:
            return None
        return _format_model_error(f"model not found: {model}", listed)

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
        that same tree. The run also executes under a throwaway `HOME`
        (`fake_home_env`) so the user's Cursor settings and skill roots never
        leak in; it authenticates from `CURSOR_API_KEY`, preserved in that
        scrubbed env. Because that throwaway `HOME` has no login keychain,
        the run also sets `AGENT_CLI_CREDENTIAL_STORE=memory` (via `cursor_env`)
        so the CLI bypasses the macOS keychain and never raises the
        blocking `cursor-user` popup. `model` is assumed to be set and is
        always forwarded as `--model`.

        A trial that errors out -- nonzero exit, an `is_error` result event,
        or a stream with no assistant events -- is retried with bounded
        back-off (`TRIAL_RETRY`) inside the one `timeout` budget; if retries
        exhaust, the returned run is marked `errored` so grading excludes it
        from the Beta-binomial counts instead of scoring the failure against
        the skill.
        """
        def attempt(remaining: float, last_error: list[str]) -> EvalRun:
            with (
                isolated_workdir(repo_root, isolate) as workdir,
                fake_home_env() as base_env,
            ):
                env = cursor_env(base_env)
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
                proc = _spawn_checked(
                    cmd, str(workdir), env, remaining, last_error
                )
                skill_invoked, assistant_text, tool_uses, run_model = (
                    parse_cursor_stream_json(proc.stdout, skill_name, workdir)
                )
            return EvalRun(
                eval_id="",
                prompt=prompt,
                skill_invoked=skill_invoked,
                assistant_text=assistant_text,
                tool_uses=tool_uses,
                model=run_model,
            )

        return _run_trial(prompt, timeout, attempt)
