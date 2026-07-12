from __future__ import annotations

from collections.abc import Callable
from contextvars import ContextVar
from dataclasses import dataclass

from binom_eval.stream_json import EvalRun


@dataclass(eq=False)
class AssertionFailure(AssertionError):
    """Structured assertion failure raised by skill assertion handlers.

    ``summary`` is one line for rollups; ``sections`` are skill-defined
    ``(label, body)`` pairs the framework renders without interpreting.
    Every assertion handler should raise this (with or without ``sections``)
    rather than plain ``AssertionError``. Prefer ``assert_check`` so the
    same ``sections`` appear in ``--live-eval-verbose`` output when a trial
    passes.
    """

    summary: str
    sections: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        super().__init__(self.summary)

    def __str__(self) -> str:
        return self.summary


@dataclass(frozen=True)
class TrialFailure:
    """Captured outcome when one trial's assertion handler fails."""

    summary: str
    sections: tuple[tuple[str, str], ...] = ()


_assertion_sections: ContextVar[tuple[tuple[str, str], ...]] = ContextVar(
    "_assertion_sections", default=()
)


def assert_check(
    ok: bool,
    summary: str,
    *,
    sections: tuple[tuple[str, str], ...] = (),
) -> None:
    """Finish an assertion handler, raising ``AssertionFailure`` on failure.

    Attach the same ``sections`` you would show on failure so
    ``--live-eval-verbose`` can render them for passing trials too.
    """
    _assertion_sections.set(sections)
    if not ok:
        raise AssertionFailure(summary, sections)


def _default_pass_sections(run: EvalRun) -> tuple[tuple[str, str], ...]:
    sections: list[tuple[str, str]] = [
        ("Assistant reply", run.assistant_text or "(empty)")
    ]
    if run.tool_uses:
        sections.append(("Tool uses", str(run.tool_uses)))
    return tuple(sections)


def evaluate_check(
    run: EvalRun, check: Callable[[EvalRun], None]
) -> tuple[bool, TrialFailure]:
    """Run ``check`` once and return ``(passed, detail)`` for verbose output."""
    token = _assertion_sections.set(())
    try:
        check(run)
        sections = _assertion_sections.get()
        if not sections:
            sections = _default_pass_sections(run)
        return True, TrialFailure("passed", sections)
    except AssertionFailure as exc:
        return False, _capture_trial_failure(exc)
    finally:
        _assertion_sections.reset(token)


def _capture_trial_failure(exc: AssertionFailure) -> TrialFailure:
    return TrialFailure(exc.summary, exc.sections)
