"""Deadline-aware HTTP retry policy with exponential back-off."""

from __future__ import annotations

import random
import time
import urllib.error
from collections.abc import Callable
from typing import Any


class HttpRetryPolicy:
    """Retry policy: exponential back-off with jitter, deadline-aware.

    Encapsulates the retry configuration (attempt count, delays, retryable
    HTTP status codes) and exposes an `execute` method that runs a callable
    with that policy applied, returning `None` on exhaustion or deadline.
    """

    def __init__(
        self,
        *,
        max_attempts: int,
        base_delay_seconds: float,
        max_delay_seconds: float,
        retryable_http: frozenset[int],
    ) -> None:
        self.max_attempts = max_attempts
        self.base_delay_seconds = base_delay_seconds
        self.max_delay_seconds = max_delay_seconds
        self.retryable_http = retryable_http

    def is_retryable(self, exc: BaseException) -> bool:
        """True when `exc` is a transient failure worth retrying."""
        if isinstance(exc, TimeoutError):
            return True
        if isinstance(exc, urllib.error.HTTPError):
            return exc.code in self.retryable_http
        return isinstance(exc, urllib.error.URLError)

    def backoff_delay(self, attempt: int) -> float:
        """Exponential back-off capped at `max_delay_seconds`, with jitter."""
        cap = min(
            self.max_delay_seconds,
            self.base_delay_seconds * (2**attempt),
        )
        return random.uniform(0, cap)

    def execute(
        self,
        fn: Callable[[float], Any],
        timeout: int,
        *,
        transport_errors: tuple[type[BaseException], ...],
        fatal_errors: tuple[type[BaseException], ...],
    ) -> Any:
        """Call ``fn(remaining_seconds)``, retrying on retryable transport errors.

        Returns ``fn``'s result on success. Returns ``None`` when the deadline
        is exceeded, all attempts are exhausted, a non-retryable transport
        error is raised, or a ``fatal_errors`` exception is raised.

        Args:
          fn: Callable that receives the remaining deadline (seconds) and
            returns a result or raises.
          timeout: Total seconds budget for all attempts combined.
          transport_errors: Exception types whose instances are passed to
            `is_retryable`; retryable ones trigger a retry, non-retryable
            ones return ``None`` immediately.
          fatal_errors: Exception types that always abort without retry.
        """
        deadline = time.monotonic() + timeout
        for attempt in range(self.max_attempts):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return None
            try:
                return fn(remaining)
            except transport_errors as exc:
                if not self.is_retryable(exc) or attempt == self.max_attempts - 1:
                    return None
            except fatal_errors:
                return None

            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return None
            time.sleep(min(self.backoff_delay(attempt), remaining))
        return None
