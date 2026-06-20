"""Invoking `claude -p` and parsing the result into `EvalRun`s.

The I/O layer of the harness: it is the only module that spawns
subprocesses and scrubs the environment. `run_claude` is a single live
call; `run_claude_batch` overlaps `count` independent calls for one eval to
measure the model's run-to-run variance. Evals are non-deterministic, so
nothing here caches — every trial is a fresh invocation.
"""

from __future__ import annotations

import os
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from binom_eval.stream_json import EvalRun, parse_stream_json

DEFAULT_TIMEOUT_SECONDS = 300


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


def run_claude(
    prompt: str,
    repo_root: Path,
    skill_name: str,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
) -> EvalRun:
    """Invoke `claude -p` once and parse its stream-json output."""
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
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=str(repo_root),
        env=stripped_env(),
        timeout=timeout,
    )
    skill_invoked, assistant_text, tool_uses = parse_stream_json(
        proc.stdout, skill_name
    )
    return EvalRun(
        eval_id="",
        prompt=prompt,
        skill_invoked=skill_invoked,
        assistant_text=assistant_text,
        tool_uses=tool_uses,
    )


def run_claude_batch(
    item: dict[str, Any],
    repo_root: Path,
    skill_name: str,
    count: int,
) -> list[EvalRun]:
    """Run `claude -p` `count` times for one eval, concurrently.

    Each run is an isolated `subprocess.run`, so threads share nothing
    mutable; concurrency just overlaps the model latency. Every trial is a
    fresh live call — repeated trials exist to measure the model's
    run-to-run variance.
    """
    eid = item["id"]
    prompt = item["prompt"]
    with ThreadPoolExecutor(max_workers=count) as pool:
        runs = list(
            pool.map(
                lambda _: run_claude(prompt, repo_root, skill_name),
                range(count),
            )
        )
    for run in runs:
        run.eval_id = eid
    return runs