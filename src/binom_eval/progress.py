from __future__ import annotations

import sys
import threading
from dataclasses import dataclass
from typing import IO, Protocol

from binom_eval.posterior import Verdict


@dataclass
class ProgressEvent:
    """Progress event during eval trials.

    kind: "batch" during trial rounds or "eval_done" after completion.
    batch_num: 1-indexed batch number, or 0 for eval_done.
    batch_size: Number of trials in current batch, or 0 for eval_done.
    trials_run: Total trials run so far (including errored).
    trials_max: Maximum trials budget for this eval.
    batch_elapsed: Seconds spent on current batch, or 0.0 for eval_done.
    total_elapsed: Seconds from eval start.
    verdict: Current verdict state (PASS, FAIL, or UNDETERMINED).
    """

    kind: str
    eval_id: str
    batch_num: int
    batch_size: int
    trials_run: int
    trials_max: int
    batch_elapsed: float
    total_elapsed: float
    verdict: Verdict


class ProgressRenderer(Protocol):
    """Protocol for rendering progress events."""

    def render(self, event: ProgressEvent) -> None:
        """Render a progress event."""
        ...


def _format_event(event: ProgressEvent) -> str:
    """Format a progress event as a human-readable string.

    Batch: [eval_id]  batch N  M/MAX trials  VERDICT  1.2s batch  4.5s total
    Done: [eval_id]  DONE  M/MAX trials  VERDICT  14.3s total
    """
    verdict_label = event.verdict.value.upper()
    trials_str = f"{event.trials_run}/{event.trials_max} trials"
    time_total = f"{event.total_elapsed:.1f}s total"

    if event.kind == "batch":
        batch_str = f"batch {event.batch_num}"
        time_batch = f"{event.batch_elapsed:.1f}s batch"
        return (
            f"[{event.eval_id}]  {batch_str}  {trials_str}  {verdict_label}  "
            f"{time_batch}  {time_total}"
        )
    else:
        return (
            f"[{event.eval_id}]  DONE  {trials_str}  {verdict_label}  "
            f"{time_total}"
        )


class TtyRenderer:
    """Progress renderer for TTY environments using carriage return overwrite."""

    def __init__(self, stream: IO[str]) -> None:
        self._stream = stream
        self._lock = threading.Lock()

    def render(self, event: ProgressEvent) -> None:
        line = _format_event(event)
        suffix = "\r" if event.kind == "batch" else "\n"
        with self._lock:
            self._stream.write(line + suffix)
            self._stream.flush()


class PlainRenderer:
    """Progress renderer for non-TTY environments using newlines."""

    def __init__(self, stream: IO[str]) -> None:
        self._stream = stream
        self._lock = threading.Lock()

    def render(self, event: ProgressEvent) -> None:
        line = _format_event(event)
        with self._lock:
            self._stream.write(line + "\n")
            self._stream.flush()


def make_renderer(stream: IO[str] | None = None) -> ProgressRenderer:
    """Create a progress renderer appropriate for the output stream.

    Uses sys.stderr if stream is None. Returns TtyRenderer if the stream
    is a TTY, PlainRenderer otherwise.
    """
    s = stream if stream is not None else sys.stderr
    if hasattr(s, "isatty") and s.isatty():
        return TtyRenderer(s)
    return PlainRenderer(s)
