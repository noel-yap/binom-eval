"""The `claude -p` backend: `ClaudeRunner`.

The concrete `Runner` for Claude Code. It owns every `claude`-specific
detail -- the binary name, the `-p` invocation flags, and the `--version`
probe -- so the rest of the harness depends only on the backend-agnostic
`Runner` interface. The shared subprocess/env helpers it builds on
(`stripped_env`, `isolated_workdir`, `_model_probe_rejected`) live in the
package root alongside the interface.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import urllib.error
import urllib.request
from pathlib import Path

from binom_eval.runner import (
    DEFAULT_TIMEOUT_SECONDS,
    Runner,
    _format_model_error,
    _model_probe_rejected,
    _run_trial,
    _spawn_checked,
    fake_home_env,
    isolated_workdir,
    stripped_env,
)
from binom_eval.runner.retry import RetryableError, RetryPolicy
from binom_eval.stream_json import EvalRun, parse_stream_json

# Live evals run under a throwaway HOME so the invoking user's skill roots never
# leak in, but still load project skills from the run's workspace via
# `--setting-sources project` (not an empty source list, which would hide them).
CLAUDE_SETTING_SOURCES = "project"

# CLI aliases accepted by `claude --model` that are not returned as API model IDs.
CLAUDE_MODEL_ALIASES = frozenset({"fable", "haiku", "opus", "sonnet"})

# HTTP statuses worth retrying: rate-limit (429) and transient upstream/server
# errors. Any other status (e.g. 401/403 auth, 404) is permanent for the
# request, so the lookup gives up and reports the model list as unavailable.
_RETRYABLE_MODELS_API_HTTP = frozenset({429, 500, 502, 503, 504})

_MODELS_API_RETRY = RetryPolicy(
    max_attempts=3,
    base_delay_seconds=0.25,
    max_delay_seconds=2.0,
)


def _parse_anthropic_models_payload(payload: object) -> list[str]:
    """Extract model IDs from a Models API response body."""
    if not isinstance(payload, dict):
        raise ValueError("unexpected models payload")
    data = payload.get("data")
    if not isinstance(data, list):
        raise ValueError("unexpected models payload")
    ids = [
        str(item["id"])
        for item in data
        if isinstance(item, dict) and item.get("id")
    ]
    if not ids:
        raise ValueError("empty models list")
    return ids


def _fetch_anthropic_model_ids_once(
    req: urllib.request.Request, timeout: float
) -> list[str] | None:
    """One Models API request, classified for `_MODELS_API_RETRY`.

    Raises `RetryableError` on a transient failure -- a timeout, a connection
    error, or a retryable HTTP status -- so the request is retried. Returns
    `None` for any permanent failure (auth error, missing or malformed
    payload) to signal that the model list is unavailable.
    """
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload = json.loads(resp.read().decode())
        return _parse_anthropic_models_payload(payload)
    except urllib.error.HTTPError as exc:
        if exc.code in _RETRYABLE_MODELS_API_HTTP:
            raise RetryableError(f"models API returned HTTP {exc.code}") from exc
        return None
    except (TimeoutError, urllib.error.URLError) as exc:
        raise RetryableError("models API request failed") from exc
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        return None


def _anthropic_model_ids(timeout: int = 30) -> list[str] | None:
    """Return model IDs from the Anthropic Models API, or None when unavailable."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return None
    req = urllib.request.Request(
        "https://api.anthropic.com/v1/models?limit=100",
        headers={
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
        },
    )
    return _MODELS_API_RETRY.execute(
        lambda remaining: _fetch_anthropic_model_ids_once(req, remaining),
        timeout,
    )


def _model_in_anthropic_list(model: str, ids: list[str]) -> bool:
    """True when `model` is a listed ID, a known CLI alias, or an ID prefix."""
    if model in CLAUDE_MODEL_ALIASES or model in ids:
        return True
    prefix = model + "-"
    return any(api_id.startswith(prefix) for api_id in ids)


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
        evals run with `--bare` and `--setting-sources project` under a
        throwaway `HOME`, so only the workspace's own skills load while user
        settings and skill roots stay hidden.
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

        Prefetches available model IDs from the Anthropic Models API and
        rejects unknown models without spawning a probe when the list is
        readable. Falls back to a cheap `claude -p` probe when the API cannot
        be reached so a flaky lookup never blocks a run that might otherwise
        succeed -- the real trials will surface any genuine outage.
        """
        listed = _anthropic_model_ids(timeout)
        if listed is not None:
            if _model_in_anthropic_list(model, listed):
                return None
            return _format_model_error(f"model not found: {model}", listed)

        cmd = [
            "claude",
            "-p",
            "ok",
            "--model",
            model,
            "--bare",
            "--setting-sources",
            CLAUDE_SETTING_SOURCES,
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
        without relying on the CLI default. The run executes under a throwaway
        `HOME` (`fake_home_env`) -- on top of `--bare --setting-sources
        project` -- so no user-level config or skill root leaks in; project
        skills from the workspace still load. The run authenticates from
        `ANTHROPIC_API_KEY`, preserved in that scrubbed env.

        A trial that errors out -- nonzero exit, an `is_error` result event
        (API 500/overload, `error_during_execution`, ...), or a stream with
        no assistant events -- is retried with bounded back-off (`TRIAL_RETRY`)
        inside the one `timeout` budget; if retries exhaust, the returned run
        is marked `errored` so grading excludes it from the Beta-binomial
        counts instead of scoring the failure against the skill.
        """
        cmd = [
            "claude",
            "-p",
            prompt,
            "--bare",
            "--setting-sources",
            CLAUDE_SETTING_SOURCES,
            "--output-format",
            "stream-json",
            "--verbose",
            "--include-partial-messages",
            "--dangerously-skip-permissions",
            "--model",
            model,
        ]
        def attempt(remaining: float, last_error: list[str]) -> EvalRun:
            with (
                isolated_workdir(repo_root, isolate) as workdir,
                fake_home_env() as env,
            ):
                proc = _spawn_checked(
                    cmd, str(workdir), env, remaining, last_error
                )
                skill_invoked, assistant_text, tool_uses, run_model = (
                    parse_stream_json(proc.stdout, skill_name, workdir)
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
