"""Invoking `claude -p` and parsing the result into `EvalRun`s.

The I/O layer of the harness: it is the only module that spawns
subprocesses and scrubs the environment. `run_claude` is a single live
call; `run_claude_batch` overlaps `count` independent calls for one eval to
measure the model's run-to-run variance. Evals are non-deterministic, so
nothing here caches — every trial is a fresh invocation.

Concurrency is throttled by an optional shared `gate` (a `threading.Semaphore`
the caller threads through every run): trials within a batch, and whole evals
running in parallel above this layer, all acquire the one gate, so total live
`claude -p` calls never exceed its count regardless of suite size. Filesystem
isolation is optional too -- with `isolate=True` each run executes in a fresh
throwaway copy of `repo_root` (see `isolated_workdir`), so a skill that writes
to the tree cannot clobber a concurrent run.
"""

from __future__ import annotations

import contextlib
import os
import shutil
import subprocess
import tempfile
import threading
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from binom_eval.stream_json import EvalRun, parse_stream_json

DEFAULT_TIMEOUT_SECONDS = 300

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


def cli_version() -> str:
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
    with tempfile.TemporaryDirectory(prefix="binom-eval-") as tmp:
        dest = Path(tmp) / repo_root.name
        shutil.copytree(
            repo_root,
            dest,
            symlinks=True,
            ignore=shutil.ignore_patterns(*ISOLATION_IGNORE),
        )
        yield dest


def run_claude(
    prompt: str,
    repo_root: Path,
    skill_name: str,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
    *,
    isolate: bool = False,
    model: str | None = None,
) -> EvalRun:
    """Invoke `claude -p` once and parse its stream-json output.

    With `isolate=True` the call runs in a throwaway copy of `repo_root`
    (see `isolated_workdir`) so a skill that writes to the tree cannot affect
    `repo_root` or a concurrent run; otherwise it runs in `repo_root` directly.
    `model`, when given, is forwarded as `--model` so callers can select a
    cheaper or faster model for eval runs without changing the CLI default.
    """
    cmd = [
        "claude",
        "-p",
        prompt,
        "--output-format",
        "stream-json",
        "--verbose",
        "--include-partial-messages",
        "--dangerously-skip-permissions",
    ]
    if model is not None:
        cmd += ["--model", model]
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


def run_claude_batch(
    item: dict[str, Any],
    repo_root: Path,
    skill_name: str,
    count: int,
    *,
    gate: threading.Semaphore | None = None,
    isolate: bool = False,
    model: str | None = None,
) -> list[EvalRun]:
    """Run `claude -p` `count` times for one eval, concurrently.

    Each run is an isolated `subprocess.run`, so threads share nothing
    mutable; concurrency just overlaps the model latency. Every trial is a
    fresh live call — repeated trials exist to measure the model's
    run-to-run variance.

    `gate`, when given, is a shared semaphore every trial acquires before
    spawning its subprocess; the same object passed across batches and across
    concurrently-running evals caps total live `claude -p` calls at its count.
    `isolate` is forwarded to `run_claude` so each trial runs in its own
    throwaway copy of `repo_root` when set. `model`, when given, selects a
    specific model for all trials in the batch.
    """
    eid = item["id"]
    prompt = item["prompt"]
    limit: contextlib.AbstractContextManager[Any] = (
        gate if gate is not None else contextlib.nullcontext()
    )

    def one(_: int) -> EvalRun:
        with limit:
            return run_claude(
                prompt, repo_root, skill_name, isolate=isolate, model=model
            )

    with ThreadPoolExecutor(max_workers=count) as pool:
        runs = list(pool.map(one, range(count)))
    for run in runs:
        run.eval_id = eid
    return runs
