"""The `claude -p` backend: `ClaudeRunner`.

The concrete `Runner` for Claude Code. It owns every `claude`-specific
detail -- the binary name, the `-p` invocation flags, and the `--version`
probe -- so the rest of the harness depends only on the backend-agnostic
`Runner` interface. The shared subprocess/env helpers it builds on
(`stripped_env`, `isolated_workdir`, `_model_probe_rejected`) live in the
package root alongside the interface.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

from binom_eval.runner import (
    DEFAULT_TIMEOUT_SECONDS,
    Runner,
    _model_probe_rejected,
    isolated_workdir,
    stripped_env,
)
from binom_eval.stream_json import EvalRun, parse_stream_json


class ClaudeRunner(Runner):
    """A `Runner` backed by the `claude -p` CLI."""

    def version(self) -> str:
        """Return the `claude` CLI version string, or '' when not on PATH."""
        try:
            result = subprocess.run(
                ["claude", "--version"],
                capture_output=True,
                text=True,
                timeout=10,
                env=stripped_env(),
            )
            return result.stdout.strip()
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return ""

    def preflight(self) -> str | None:
        """Return why `claude -p` cannot run, or None when ready.

        Requires the `claude` CLI on PATH and `ANTHROPIC_API_KEY` set: live
        evals run with `--bare` (no settings sources), so the key is the only
        credential the nested run can authenticate with.
        """
        if shutil.which("claude") is None:
            return "claude CLI not found on PATH"
        if not os.environ.get("ANTHROPIC_API_KEY"):
            return (
                "ANTHROPIC_API_KEY is not set; live evals run with isolated "
                "settings (`--bare`) and authenticate only via that key."
            )
        return None

    def validate_model(self, model: str, timeout: int = 30) -> str | None:
        """Confirm the `claude` CLI can use `model`; return an error or None.

        Runs one trivial `claude -p` probe with `--model model`. Returns None
        when the model is usable, or the CLI's error message when the model
        does not exist or the account cannot access it (the CLI reports
        `model_not_found` / HTTP 404). The probe is cheap: a bad model is
        rejected in ~1s at zero cost, a good one costs a single short turn.
        Transport failures (CLI absent, timeout) return None so a flaky probe
        never blocks a run that might otherwise succeed -- the real trials will
        surface any genuine outage.
        """
        cmd = [
            "claude",
            "-p",
            "ok",
            "--model",
            model,
            "--bare",
            "--setting-sources",
            "",
            "--output-format",
            "stream-json",
            "--verbose",
            "--dangerously-skip-permissions",
        ]
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                env=stripped_env(),
                timeout=timeout,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return None
        return _model_probe_rejected(proc.stdout)

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
        """Invoke `claude -p` once and parse its stream-json output.

        With `isolate=True` the call runs in a throwaway copy of `repo_root`
        (see `isolated_workdir`) so a skill that writes to the tree cannot
        affect `repo_root` or a concurrent run; otherwise it runs in
        `repo_root` directly. `model` is assumed to be set and is always
        forwarded as `--model` so callers select a specific model for eval runs
        without relying on the CLI default.
        """
        cmd = [
            "claude",
            "-p",
            prompt,
            "--bare",
            "--setting-sources",
            "",
            "--output-format",
            "stream-json",
            "--verbose",
            "--include-partial-messages",
            "--dangerously-skip-permissions",
            "--model",
            model,
        ]
        with isolated_workdir(repo_root, isolate) as workdir:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=str(workdir),
                env=stripped_env(),
                timeout=timeout,
            )
        skill_invoked, assistant_text, tool_uses, model = parse_stream_json(
            proc.stdout, skill_name
        )
        return EvalRun(
            eval_id="",
            prompt=prompt,
            skill_invoked=skill_invoked,
            assistant_text=assistant_text,
            tool_uses=tool_uses,
            model=model,
        )
