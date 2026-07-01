"""Unit tests for `binom_eval.runner.retry.HttpRetryPolicy`."""

from __future__ import annotations

import io
import urllib.error

import pytest

from binom_eval.runner import retry as runner_retry
from binom_eval.runner.retry import HttpRetryPolicy


def _policy(**overrides: object) -> HttpRetryPolicy:
    """Return a policy with sensible defaults, overridable per-test."""
    return HttpRetryPolicy(
        max_attempts=int(overrides.get("max_attempts", 3)),
        base_delay_seconds=float(overrides.get("base_delay_seconds", 0.25)),
        max_delay_seconds=float(overrides.get("max_delay_seconds", 2.0)),
        retryable_http=frozenset(overrides.get("retryable_http", {503})),
    )


def _http_error(code: int) -> urllib.error.HTTPError:
    return urllib.error.HTTPError(
        "http://example.com", code, "err", hdrs=None, fp=io.BytesIO(b"")
    )


_TRANSPORT = (urllib.error.URLError,)
_FATAL = (ValueError,)


class TestIsRetryable:
    def test_timeout_error_returns_true(self) -> None:
        assert _policy().is_retryable(TimeoutError())

    def test_http_error_with_listed_code_returns_true(self) -> None:
        assert _policy(retryable_http={503}).is_retryable(_http_error(503))

    def test_http_error_with_unlisted_code_returns_false(self) -> None:
        assert not _policy(retryable_http={503}).is_retryable(_http_error(401))

    def test_plain_url_error_returns_true(self) -> None:
        assert _policy().is_retryable(urllib.error.URLError("network down"))

    def test_unrelated_exception_returns_false(self) -> None:
        assert not _policy().is_retryable(ValueError("bad"))


class TestExecute:
    def test_returns_result_when_fn_succeeds_on_first_attempt(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(runner_retry.time, "sleep", lambda _: None)
        result = _policy().execute(
            lambda _r: "ok",
            timeout=30,
            transport_errors=_TRANSPORT,
            fatal_errors=_FATAL,
        )
        assert result == "ok"

    def test_returns_none_when_deadline_already_expired_at_loop_start(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # monotonic() returns: [set_deadline=0, first_remaining_check=9999]
        ticks = iter([0.0, 9999.0])
        monkeypatch.setattr(runner_retry.time, "monotonic", lambda: next(ticks))
        monkeypatch.setattr(runner_retry.time, "sleep", lambda _: None)
        result = _policy().execute(
            lambda _r: "never",
            timeout=30,
            transport_errors=_TRANSPORT,
            fatal_errors=_FATAL,
        )
        assert result is None

    def test_returns_none_immediately_on_non_retryable_transport_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(runner_retry.time, "sleep", lambda _: None)
        calls = {"n": 0}

        def fn(_r: float) -> str:
            calls["n"] += 1
            raise _http_error(401)

        result = _policy(max_attempts=3, retryable_http={503}).execute(
            fn, timeout=30, transport_errors=_TRANSPORT, fatal_errors=_FATAL
        )
        assert result is None
        assert calls["n"] == 1

    def test_returns_none_after_all_retryable_attempts_exhausted(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(runner_retry.time, "sleep", lambda _: None)
        calls = {"n": 0}

        def fn(_r: float) -> str:
            calls["n"] += 1
            raise urllib.error.URLError("transient")

        result = _policy(max_attempts=3).execute(
            fn, timeout=30, transport_errors=_TRANSPORT, fatal_errors=_FATAL
        )
        assert result is None
        assert calls["n"] == 3

    def test_retries_and_returns_result_when_fn_eventually_succeeds(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(runner_retry.time, "sleep", lambda _: None)
        calls = {"n": 0}

        def fn(_r: float) -> str:
            calls["n"] += 1
            if calls["n"] < 3:
                raise urllib.error.URLError("transient")
            return "success"

        result = _policy(max_attempts=3).execute(
            fn, timeout=30, transport_errors=_TRANSPORT, fatal_errors=_FATAL
        )
        assert result == "success"
        assert calls["n"] == 3

    def test_returns_none_immediately_on_fatal_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(runner_retry.time, "sleep", lambda _: None)
        calls = {"n": 0}

        def fn(_r: float) -> str:
            calls["n"] += 1
            raise ValueError("bad payload")

        result = _policy(max_attempts=3).execute(
            fn, timeout=30, transport_errors=_TRANSPORT, fatal_errors=_FATAL
        )
        assert result is None
        assert calls["n"] == 1

    def test_returns_none_when_deadline_expires_after_failed_attempt(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # monotonic() returns: [set_deadline=0, first_remaining=1, post_fail_remaining=9999]
        ticks = iter([0.0, 1.0, 9999.0])
        monkeypatch.setattr(runner_retry.time, "monotonic", lambda: next(ticks))
        monkeypatch.setattr(runner_retry.time, "sleep", lambda _: None)

        def fn(_r: float) -> str:
            raise urllib.error.URLError("transient")

        result = _policy(max_attempts=3).execute(
            fn, timeout=30, transport_errors=_TRANSPORT, fatal_errors=_FATAL
        )
        assert result is None

    def test_passes_remaining_time_to_fn(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # deadline = 0 + 30 = 30; remaining at call = 30 - 5 = 25
        ticks = iter([0.0, 5.0])
        monkeypatch.setattr(runner_retry.time, "monotonic", lambda: next(ticks))
        monkeypatch.setattr(runner_retry.time, "sleep", lambda _: None)
        received: dict[str, float] = {}

        def fn(remaining: float) -> str:
            received["remaining"] = remaining
            return "done"

        _policy().execute(
            fn, timeout=30, transport_errors=_TRANSPORT, fatal_errors=_FATAL
        )
        assert received["remaining"] == pytest.approx(25.0)

    def test_sleeps_between_retries(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(runner_retry.random, "uniform", lambda _a, b: b)
        slept: list[float] = []
        monkeypatch.setattr(runner_retry.time, "sleep", slept.append)
        calls = {"n": 0}

        def fn(_r: float) -> str:
            calls["n"] += 1
            if calls["n"] < 2:
                raise urllib.error.URLError("transient")
            return "ok"

        _policy(max_attempts=3).execute(
            fn, timeout=30, transport_errors=_TRANSPORT, fatal_errors=_FATAL
        )
        assert len(slept) == 1
        assert slept[0] > 0
