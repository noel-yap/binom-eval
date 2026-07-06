"""Invoking an agent CLI and parsing the result into `EvalRun`s.

The I/O layer of the harness: it is the only package that spawns
subprocesses and scrubs the environment. A `Runner` is a single live call;
`run_claude_batch` overlaps `count` independent calls for one eval to
measure the model's run-to-run variance. Evals are non-deterministic, so
nothing here caches — every trial is a fresh invocation.

Backends are pluggable: `Runner` is the backend-agnostic interface and
`ClaudeRunner` (in `claude_runner.py`) is the `claude -p` implementation.
The shared subprocess/env helpers (`stripped_env`, `fake_home_env`,
`isolated_workdir`, `_model_probe_rejected`) live here so every backend can
build on them. Live runs execute under a throwaway `HOME` (`fake_home_env`)
so no harness picks up the invoking user's settings -- user skill roots
(`~/.claude/skills`, `~/.cursor/skills`, ...), MCP config, or stored logins --
and grade only against the project; backends therefore authenticate from
environment credentials (`ANTHROPIC_API_KEY`, `CURSOR_API_KEY`) rather than a
stored session under the real home.

Concurrency is throttled by an optional shared `gate` (a `threading.Semaphore`
the caller threads through every run): trials within a batch, and whole evals
running in parallel above this layer, all acquire the one gate, so total live
calls never exceed its count regardless of suite size. Filesystem isolation is
optional too -- with `isolate=True` each run executes in a fresh throwaway copy
of `repo_root` (see `isolated_workdir`), so a skill that writes to the tree
cannot clobber a concurrent run.
"""

from __future__ import annotations

import contextlib
import json
import os
import shutil
import subprocess
import tempfile
import threading
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from binom_eval.runner.retry import RetryableError, RetryPolicy
from binom_eval.stream_json import EvalRun, parse_stream_json, stream_error

DEFAULT_TIMEOUT_SECONDS = 300

# Back-off for one live trial. An API 500/overload or a CLI that dies
# mid-run surfaces as a nonzero exit or an `is_error` result event; grading
# such a trial as a behavioral failure would bias every posterior downward
# with noise unrelated to the skill, so backends retry the trial a few times
# (within the trial's own timeout budget) before marking the run errored.
# `_spawn_checked` raises every transient trial failure as `RetryableError`,
# which is exactly what `RetryPolicy` retries.
TRIAL_RETRY = RetryPolicy(
    max_attempts=3,
    base_delay_seconds=1.0,
    max_delay_seconds=8.0,
)

# Regenerable or heavy directories not copied into a per-run isolated
# workdir: caches are rebuilt on demand and dependency trees would dominate
# the per-run copy cost. `.git` is deliberately kept so skills that shell out
# to git still see a real repository.
ISOLATION_IGNORE = (
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".venv",
    "node_modules",
)


# Markers Claude Code sets on every child process to signal a nested
# session. The CLI itself strips this exact trio when it needs a child to
# behave like a clean top-level invocation, so the eval runner mirrors it.
NESTED_SESSION_MARKERS = (
    "CLAUDECODE",
    "CLAUDE_CODE_SESSION_ID",
    "CLAUDE_CODE_CHILD_SESSION",
)


def stripped_env() -> dict[str, str]:
    """A copy of the current environment with nested-session markers removed.

    The nested `claude -p` runs must not inherit the outer session's
    `CLAUDECODE`, `CLAUDE_CODE_SESSION_ID`, or `CLAUDE_CODE_CHILD_SESSION`
    markers, which would otherwise make the CLI behave as a nested session
    rather than a fresh top-level invocation.
    """
    return {
        k: v for k, v in os.environ.items() if k not in NESTED_SESSION_MARKERS
    }


@contextlib.contextmanager
def fake_home_env() -> Iterator[dict[str, str]]:
    """Yield a scrubbed env whose `HOME` points at a fresh empty directory.

    Layered on `stripped_env`, this repoints `HOME` (and `USERPROFILE` on
    Windows) at a throwaway temp directory for the duration of one run, so a
    spawned agent CLI cannot read the invoking user's home: no user skill
    roots (`~/.claude/skills`, `~/.cursor/skills`, ...), no per-user MCP or CLI
    config, and no stored login session. Evals therefore grade only against
    the project's own skills/agents, never whatever happens to be installed
    for the user. Because the stored session is hidden, every backend must
    authenticate from an environment credential (e.g. `ANTHROPIC_API_KEY`,
    `CURSOR_API_KEY`), which is preserved by `stripped_env`. The temp home is
    removed when the run ends.
    """
    env = stripped_env()
    with tempfile.TemporaryDirectory(prefix="binom-eval-home-", ignore_cleanup_errors=True) as home:
        env["HOME"] = home
        env["USERPROFILE"] = home
        yield env


def _model_probe_rejected(stdout: str) -> str | None:
    """Verdict for a model probe's stream-json `stdout`, with no I/O.

    Returns the CLI's human-readable error message when the run reports the
    model is unusable -- `error == "model_not_found"` on any event, or an
    `is_error` result carrying HTTP 404 -- and None otherwise. Kept pure so
    the parsing is unit-testable without spawning `claude`.
    """
    message: str | None = None
    rejected = False
    for line in stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if event.get("error") == "model_not_found":
            rejected = True
        if event.get("type") == "result" and event.get("is_error"):
            if event.get("api_error_status") == 404:
                rejected = True
            message = event.get("result") or message
    return (message or "model not found") if rejected else None


def _run_error(
    returncode: int | None, stdout: str, stderr: str = ""
) -> str | None:
    """Why one completed CLI trial should not be graded, or None when clean.

    A nonzero exit means the CLI itself failed (the reason usually lands on
    stderr); a clean exit can still carry an errored stream (`is_error`
    result event, or no assistant events at all) -- see `stream_error`.
    """
    if returncode:
        lines = (stderr or stdout).strip().splitlines()
        tail = lines[-1] if lines else ""
        message = f"CLI exited with status {returncode}"
        return f"{message}: {tail}" if tail else message
    return stream_error(stdout)


def _errored_run(prompt: str, error: str) -> EvalRun:
    """An `EvalRun` marking a trial that produced no gradable result."""
    return EvalRun(
        eval_id="",
        prompt=prompt,
        skill_invoked=False,
        assistant_text="",
        tool_uses=[],
        model="",
        errored=True,
        error=error,
    )


def _spawn_checked(
    cmd: list[str],
    cwd: str,
    env: dict[str, str],
    remaining: float,
    last_error: list[str],
) -> subprocess.CompletedProcess[str]:
    """Run one CLI trial attempt; raise `RetryableError` on any error signal.

    Converts the error outcomes -- `subprocess.TimeoutExpired`, nonzero exit,
    or an errored stream (see `_run_error`) -- into `RetryableError` so the
    `TRIAL_RETRY` loop in `_run_trial` retries them, recording the reason in
    `last_error` for the eventual errored run.
    """
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=cwd,
            env=env,
            timeout=remaining,
        )
    except subprocess.TimeoutExpired:
        last_error[0] = f"trial timed out after {remaining:.0f}s"
        raise RetryableError(last_error[0]) from None
    error = _run_error(proc.returncode, proc.stdout, proc.stderr)
    if error is not None:
        last_error[0] = error
        raise RetryableError(error)
    return proc


def _run_trial(
    prompt: str,
    timeout: int,
    attempt: Callable[[float, list[str]], EvalRun],
) -> EvalRun:
    """Drive one trial's attempts through `TRIAL_RETRY`.

    `attempt(remaining_seconds, last_error)` performs a single live call
    (typically via `_spawn_checked`, which records its failure reason in
    `last_error` before raising `RetryableError`). Returns the first
    successful attempt's run, or an errored `EvalRun` carrying the last
    failure reason once retries or the `timeout` budget are exhausted.
    """
    last_error = ["trial produced no result"]
    result = TRIAL_RETRY.execute(
        lambda remaining: attempt(remaining, last_error),
        timeout,
    )
    if result is None:
        return _errored_run(prompt, last_error[0])
    return result


def _format_model_error(
    message: str, valid_models: list[str] | None = None
) -> str:
    """Append a sorted valid-models suffix when the backend could enumerate them."""
    if not valid_models:
        return message
    known = ", ".join(sorted(set(valid_models)))
    return f"{message}; valid models: {known}"


@contextlib.contextmanager
def isolated_workdir(repo_root: Path, isolate: bool) -> Iterator[Path]:
    """Yield the working directory for a single run.

    When `isolate` is false, yields `repo_root` unchanged -- every run shares
    the one tree, which is safe only for skills that do not write to it. When
    true, copies `repo_root` into a fresh temporary directory (skipping the
    regenerable/heavy entries in `ISOLATION_IGNORE`) and yields that copy, so
    a skill that mutates the tree cannot clobber a concurrent run; the copy is
    removed when the run ends.

    Args:
      repo_root: The tree `claude -p` should run against.
      isolate: Whether to run in a throwaway copy rather than `repo_root`.

    Yields:
      The directory to use as the run's `cwd`.
    """
    if not isolate:
        yield repo_root
        return
    with tempfile.TemporaryDirectory(prefix="binom-eval-", ignore_cleanup_errors=True) as tmp:
        dest = Path(tmp) / repo_root.name
        shutil.copytree(
            repo_root,
            dest,
            symlinks=True,
            ignore=shutil.ignore_patterns(*ISOLATION_IGNORE),
        )
        yield dest


class Runner(ABC):
    """A backend that can probe and invoke an agent CLI for one eval run.

    The harness depends only on this interface, so a suite can be graded
    against `claude -p`, `cursor`, or any other agent CLI by swapping in a
    different implementation. Concrete runners own the CLI specifics (binary
    name, flags, version probe); everything above this layer is backend-
    agnostic.
    """

    @abstractmethod
    def version(self) -> str:
        """Return the backend CLI version string, or '' when unavailable."""

    @abstractmethod
    def preflight(self) -> str | None:
        """Return why this backend cannot run live evals, or None when ready.

        Checked once before any trial so a missing CLI or absent credential
        fails the session fast with a clear message rather than surfacing as a
        run-time error on every trial. The message is backend-specific (e.g.
        which binary must be on PATH, which credential must be set).
        """

    @abstractmethod
    def validate_model(self, model: str, timeout: int = 30) -> str | None:
        """Confirm the backend can use `model`; return an error or None."""

    @abstractmethod
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
        """Invoke the backend once and parse its output into an `EvalRun`."""


# Imported after `Runner` and the shared helpers are defined: the backend
# modules import them back from this package, so the names must already be
# bound when their module bodies execute.
from binom_eval.runner.claude_runner import ClaudeRunner  # noqa: E402
from binom_eval.runner.cursor_runner import CursorRunner  # noqa: E402

# The selectable backends, keyed by the prefix used in `--live-eval-model`
# (`backend:model`). The prefix is mandatory -- there is no default backend --
# so every live run names the harness it targets.
BACKENDS: dict[str, type[Runner]] = {
    "claude": ClaudeRunner,
    "cursor": CursorRunner,
}


def resolve_runner(spec: str | None) -> tuple[str, str, Runner]:
    """Parse a `--live-eval-model` spec into (backend, model, runner).

    The spec must be `backend:model` (e.g. `claude:haiku` or
    `cursor:sonnet-4.5`): the backend is always explicit so each run targets a
    single, named harness. The split is on the first colon only, so model
    names may themselves contain colons.

    Raises:
      ValueError: when the spec is missing, carries no `backend:` prefix, names
        an unknown backend, or has an empty model -- callers surface this as a
        clear command-line error.
    """
    known = ", ".join(sorted(BACKENDS))
    backend, sep, model = (spec or "").partition(":")
    if not sep:
        raise ValueError(
            "--live-eval-model must be 'backend:model' "
            f"(known backends: {known}); got {spec!r}"
        )
    if backend not in BACKENDS:
        raise ValueError(
            f"unknown eval backend {backend!r} in --live-eval-model {spec!r}; "
            f"known backends: {known}"
        )
    if not model:
        raise ValueError(f"--live-eval-model {spec!r} has an empty model")
    return backend, model, BACKENDS[backend]()


# Shared default backend and thin module-level shims. The `claude -p` logic
# now lives on `ClaudeRunner`; these wrappers preserve the historical
# function-level API so existing callers keep working.
_default_runner: Runner = ClaudeRunner()


def cli_version() -> str:
    """Return the `claude` CLI version string, or '' when not on PATH."""
    return _default_runner.version()


def validate_model(model: str, timeout: int = 30) -> str | None:
    """Confirm the `claude` CLI can use `model`; return an error or None."""
    return _default_runner.validate_model(model, timeout)


def run_claude(
    prompt: str,
    repo_root: Path,
    skill_name: str,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
    *,
    isolate: bool = False,
    model: str,
) -> EvalRun:
    """Invoke `claude -p` once and parse its stream-json output."""
    return _default_runner.run(
        prompt,
        repo_root,
        skill_name,
        timeout,
        isolate=isolate,
        model=model,
    )


def run_claude_batch(
    item: dict[str, Any],
    repo_root: Path,
    skill_name: str,
    count: int,
    *,
    gate: threading.Semaphore | None = None,
    isolate: bool = False,
    model: str,
    runner: Runner | None = None,
) -> list[EvalRun]:
    """Run one eval `count` times against `runner`, concurrently.

    Each run is an isolated `subprocess.run`, so threads share nothing
    mutable; concurrency just overlaps the model latency. Every trial is a
    fresh live call — repeated trials exist to measure the model's
    run-to-run variance.

    `gate`, when given, is a shared semaphore every trial acquires before
    spawning its subprocess; the same object passed across batches and across
    concurrently-running evals caps total live calls at its count. `isolate`
    is forwarded to the runner so each trial runs in its own throwaway copy of
    `repo_root` when set. `model` selects the specific model used for all
    trials in the batch. `runner` selects the backend; it defaults to the
    shared `claude -p` runner so historical callers keep working.
    """
    backend = runner if runner is not None else _default_runner
    eid = item["id"]
    prompt = item["prompt"]
    prompt_input = item.get("prompt_input", "")
    limit: contextlib.AbstractContextManager[Any] = (
        gate if gate is not None else contextlib.nullcontext()
    )

    def one(_: int) -> EvalRun:
        with limit:
            return backend.run(
                prompt, repo_root, skill_name, isolate=isolate, model=model
            )

    with ThreadPoolExecutor(max_workers=count) as pool:
        runs = list(pool.map(one, range(count)))
    for run in runs:
        run.eval_id = eid
        run.prompt_input = prompt_input
    return runs


__all__ = [
    "BACKENDS",
    "DEFAULT_TIMEOUT_SECONDS",
    "ISOLATION_IGNORE",
    "NESTED_SESSION_MARKERS",
    "TRIAL_RETRY",
    "ClaudeRunner",
    "CursorRunner",
    "Runner",
    "cli_version",
    "fake_home_env",
    "isolated_workdir",
    "resolve_runner",
    "run_claude",
    "run_claude_batch",
    "stripped_env",
    "validate_model",
]
