"""Deadline-aware retry policy with exponential back-off."""

from __future__ import annotations

import random
import time
from collections.abc import Callable
from typing import Any


class RetryableError(Exception):
    """A failure the raiser has judged transient and worth retrying.

    Callers translate their own failure signals into this exception -- a CLI
    trial that exited nonzero or streamed an `is_error` result, or a Models
    API request that returned a retryable HTTP status -- so `RetryPolicy`
    retries them. Any other exception aborts the retry loop.
    """


class RetryPolicy:
    """Retry policy: exponential back-off with jitter, deadline-aware.

    Runs a callable, retrying it while it raises `RetryableError`, within a
    single total time budget. Returns the callable's result on success, or
    `None` once the attempts or the deadline are exhausted.
    """

    def __init__(
        self,
        *,
        max_attempts: int,
        base_delay_seconds: float,
        max_delay_seconds: float,
    ) -> None:
        self.max_attempts = max_attempts
        self.base_delay_seconds = base_delay_seconds
        self.max_delay_seconds = max_delay_seconds

    def backoff_delay(self, attempt: int) -> float:
        """Exponential back-off capped at `max_delay_seconds`, with jitter."""
        cap = min(
            self.max_delay_seconds,
            self.base_delay_seconds * (2**attempt),
        )
        return random.uniform(0, cap)

    def execute(self, fn: Callable[[float], Any], timeout: int) -> Any:
        """Call ``fn(remaining_seconds)``, retrying while it raises `RetryableError`.

        Returns ``fn``'s result on success. Returns ``None`` when the deadline
        is exceeded or all attempts are exhausted. Any exception other than
        `RetryableError` propagates to the caller, so callers translate the
        failures they want retried into `RetryableError` and handle the rest
        themselves (e.g. by returning a sentinel result).

        Args:
          fn: Callable that receives the remaining deadline (seconds) and
            returns a result or raises.
          timeout: Total seconds budget for all attempts combined.
        """
        deadline = time.monotonic() + timeout
        for attempt in range(self.max_attempts):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return None
            try:
                return fn(remaining)
            except RetryableError:
                if attempt == self.max_attempts - 1:
                    return None

            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return None
            time.sleep(min(self.backoff_delay(attempt), remaining))
        return None
