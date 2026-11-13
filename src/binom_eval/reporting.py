from __future__ import annotations

from collections.abc import Callable
from typing import Any

from binom_eval.stream_json import EvalRun
from binom_eval.posterior import (
    PASS_THRESHOLD,
    eval_passed,
    max_target_at_pass_threshold,
    posterior_pass_prob,
)
from binom_eval.planning import implied_delta
from binom_eval.assertions import (
    AssertionFailure,
    TrialFailure,
    _capture_trial_failure,
    evaluate_check,
)

# Per-section cap when rendering structured trial failures in pytest output.
# Overridable per run via `--live-eval-failure-max-chars`; zero or negative
# disables truncation.
FAILURE_SECTION_MAX_CHARS = 2000


def graded_runs(runs: list[EvalRun]) -> list[EvalRun]:
    """The trials that count toward the Beta-binomial posterior.

    Errored trials (CLI died, API error, retries exhausted -- see
    `EvalRun.errored`) are excluded everywhere a pass/fail count is taken:
    an infrastructure failure carries no evidence about the skill's true
    pass rate, so grading it as a behavioral failure would bias the
    posterior downward with noise. Errored trials still count against the
    `MAX_TRIALS` budget (their cost was spent).
    """
    return [run for run in runs if not run.errored]


def _truncate_section_body(
    body: str, max_chars: int = FAILURE_SECTION_MAX_CHARS
) -> str:
    if max_chars <= 0 or len(body) <= max_chars:
        return body
    omitted = len(body) - max_chars
    return f"{body[:max_chars]}\n... ({omitted} chars truncated)"


def _format_trial_failure(
    idx: int,
    failure: TrialFailure,
    max_chars: int = FAILURE_SECTION_MAX_CHARS,
) -> str:
    lines = [f"  trial {idx}: {failure.summary}"]
    for label, body in failure.sections:
        lines.append(f"    {label}:")
        for line in _truncate_section_body(body, max_chars).splitlines():
            lines.append(f"      {line}")
    return "\n".join(lines)


def _verbose_trials_detail(
    runs: list[EvalRun],
    outcomes: list[tuple[int, TrialFailure | None]],
    check: Callable[[EvalRun], None],
    max_chars: int,
) -> str:
    """Render every gradable trial, pass or fail, one block per trial."""
    lines: list[str] = []
    for idx, err in outcomes:
        detail = err
        if detail is None:
            _, detail = evaluate_check(runs[idx], check)
        lines.append(_format_trial_failure(idx, detail, max_chars))
    if not lines:
        return "  (no gradable trials: every trial errored)"
    return "\n".join(lines)


def trial_outcomes(
    runs: list[EvalRun], check: Callable[[EvalRun], None]
) -> list[tuple[int, TrialFailure | None]]:
    """Run `check` against each gradable trial run, capturing its result.

    `check` is an assertion handler that raises ``AssertionFailure`` on
    failure. Returns one ``(trial_index, failure_or_None)`` per graded run,
    where ``None`` means that trial passed. Errored trials are skipped (see
    `graded_runs`); indices still refer to positions in ``runs`` so a trial
    can be cross-referenced against the full batch.
    """
    outcomes: list[tuple[int, TrialFailure | None]] = []
    for idx, run in enumerate(runs):
        if run.errored:
            continue
        try:
            check(run)
            outcomes.append((idx, None))
        except AssertionFailure as exc:
            outcomes.append((idx, _capture_trial_failure(exc)))
    return outcomes


def trial_outcomes_passed(
    outcomes: list[tuple[int, str | None]],
    target: float,
    *,
    pass_threshold: float = PASS_THRESHOLD,
) -> bool:
    """True when trial outcomes clear the posterior bar at ``target``."""
    passes = sum(1 for _, err in outcomes if err is None)
    return eval_passed(
        passes, len(outcomes), target, pass_threshold=pass_threshold
    )


def format_posterior_summary(
    label: str,
    passes: int,
    trials: int,
    target: float,
    *,
    pass_threshold: float = PASS_THRESHOLD,
) -> str:
    """One-line posterior calibration summary for pytest output.

    Reports ``P(θ ≥ θ₀ | k, n)`` at the configured target ``θ₀``, and the
    highest target ``max θ₀`` that still PASS-locks at ``τ = pass_threshold``
    given the same ``k`` and ``n``.
    """
    p_good = posterior_pass_prob(passes, trials, target)
    max_target = max_target_at_pass_threshold(passes, trials, pass_threshold)
    return (
        f"{label}: {passes}/{trials} trials passed; "
        f"P(θ ≥ {target:.3f} | k={passes}, n={trials}) = {p_good:.3f}; "
        f"max θ₀ (pass@τ={pass_threshold:.3f} | k={passes}, n={trials}) = "
        f"{max_target:.3f}"
    )


def weakest_k_n(counts: list[tuple[int, int]]) -> tuple[int, int]:
    """``(k, n)`` of the assertion with the lowest ``k/n``.

    An eval passes only when every assertion does, and ``p_good`` rises with
    ``k``, so the weakest assertion decides the eval's grade and its
    ``(k, n)`` stands for the whole eval.
    """
    weakest = counts[0]
    for k, n in counts[1:]:
        if k * weakest[1] < weakest[0] * n:
            weakest = (k, n)
    return weakest


def format_trials_histogram(
    trials: list[int], *, width: int = 30, max_rows: int = 12
) -> list[str]:
    """Bar chart of how many evals ran each number of trials.

    One row per distinct ``n``; when more than ``max_rows`` distinct values
    occur, ``n`` is grouped into ``max_rows`` equal-width ranges.
    """
    if not trials:
        return []
    distinct = sorted(set(trials))
    if len(distinct) <= max_rows:
        buckets = [(n, n) for n in distinct]
    else:
        lo, hi = distinct[0], distinct[-1]
        step = -(-(hi - lo + 1) // max_rows)
        buckets = [
            (lo + i * step, min(hi, lo + (i + 1) * step - 1))
            for i in range(max_rows)
        ]
    counts = [sum(1 for t in trials if a <= t <= b) for a, b in buckets]
    peak = max(counts)
    labels = [str(a) if a == b else f"{a}-{b}" for a, b in buckets]
    label_w = max(len(label) for label in labels)
    ordered = sorted(trials)
    median = ordered[len(ordered) // 2]
    lines = ["trials per eval (distribution across evals):"]
    for label, count in zip(labels, counts):
        bar = "█" * max(1 if count else 0, round(width * count / peak))
        lines.append(f"  n={label:>{label_w}}  {bar} {count}")
    lines.append(
        f"  evals: {len(trials)};  trials min {ordered[0]}, "
        f"median {median}, max {ordered[-1]}, total {sum(trials)}"
    )
    return lines


def format_k_n_header(target: float, pass_threshold: float) -> str:
    """Legend line printed once above the per-eval k/n bars."""
    return (
        f"k/n per eval, weakest assertion  (█ = k/n, | = θ₀ {target:.3f}, "
        f"τ = {pass_threshold:.4f}):"
    )


def format_k_n_bar(
    label: str,
    passes: int,
    trials: int,
    target: float,
    pass_threshold: float = PASS_THRESHOLD,
    *,
    width: int = 30,
) -> str:
    """One bar row: ``k/n`` out of ``[0, 1]`` with ``|`` marking ``θ₀``, the
    PASS/FAIL grade at ``τ`` and the ``δ`` that ``trials`` resolve
    (``implied_delta``).
    """
    frac = passes / trials if trials else 0.0
    filled = round(width * frac)
    marker = min(width - 1, round(width * target))
    cells = ["█" if i < filled else "░" for i in range(width)]
    cells[marker] = "|" if marker >= filled else "▐"
    ok = eval_passed(passes, trials, target, pass_threshold=pass_threshold)
    delta = implied_delta(trials, target, pass_threshold)
    delta_text = "n/a" if delta is None else f"{delta:.3f}"
    return (
        f"  {label}  {''.join(cells)}  {passes:>3}/{trials:<3} {frac:.3f}  "
        f"{'PASS' if ok else 'FAIL'}  δ={delta_text}"
    )


def format_k_n_summary(
    results: list[tuple[str, int, int]],
    target: float,
    pass_threshold: float = PASS_THRESHOLD,
    *,
    planned_delta: float | None = None,
) -> list[str]:
    """Aggregate θ₀/δ/τ footer over ``(eval, passes, trials)`` results."""
    if not results:
        return []
    passed = sum(
        eval_passed(k, n, target, pass_threshold=pass_threshold)
        for _, k, n in results
    )
    total_k = sum(k for _, k, _ in results)
    total_n = sum(n for _, _, n in results)
    summary = (
        f"  evals: {passed}/{len(results)} PASS;  "
        f"pooled k/n = {total_k}/{total_n} "
        f"({total_k / total_n if total_n else 0.0:.3f});  "
        f"θ₀ = {target:.3f};  τ = {pass_threshold:.4f}"
    )
    if planned_delta is not None:
        summary += f";  planned δ = {planned_delta:.3f}"
    lines = [summary]
    deltas = [
        d
        for _, _, n in results
        if (d := implied_delta(n, target, pass_threshold)) is not None
    ]
    if deltas:
        lines.append(
            f"  implied δ: min {min(deltas):.3f}, "
            f"mean {sum(deltas) / len(deltas):.3f}, max {max(deltas):.3f}"
        )
    return lines


def trial_outcomes_posterior_summary(
    outcomes: list[tuple[int, TrialFailure | None]],
    target: float,
    label: str,
    *,
    pass_threshold: float = PASS_THRESHOLD,
) -> str:
    """``format_posterior_summary`` for a ``trial_outcomes`` result list."""
    passes = sum(1 for _, err in outcomes if err is None)
    return format_posterior_summary(
        label, passes, len(outcomes), target, pass_threshold=pass_threshold
    )


def trial_outcomes_verbose_message(
    runs: list[EvalRun],
    outcomes: list[tuple[int, TrialFailure | None]],
    check: Callable[[EvalRun], None],
    target: float,
    label: str,
    *,
    pass_threshold: float = PASS_THRESHOLD,
    max_chars: int = FAILURE_SECTION_MAX_CHARS,
) -> str:
    """Human-readable pass detail for ``trial_outcomes_passed``.

    Renders every gradable trial with the same section layout as
    ``trial_outcomes_failure_message``. Passing trials reuse the handler's
    ``assert_check`` sections; failing trials reuse the captured failure detail.
    """
    passes = sum(1 for _, err in outcomes if err is None)
    trials = len(outcomes)
    return (
        format_posterior_summary(
            label, passes, trials, target, pass_threshold=pass_threshold
        )
        + "\nTrials:\n"
        + _verbose_trials_detail(runs, outcomes, check, max_chars)
    )


def graded_runs_verbose_message(
    runs: list[EvalRun],
    label: str,
    passes: int,
    trials: int,
    target: float,
    check: Callable[[EvalRun], None],
    *,
    pass_threshold: float = PASS_THRESHOLD,
    max_chars: int = FAILURE_SECTION_MAX_CHARS,
) -> str:
    """Verbose trial listing for count-based rollups (e.g. trigger checks)."""
    outcomes = trial_outcomes(runs, check)
    return (
        format_posterior_summary(
            label, passes, trials, target, pass_threshold=pass_threshold
        )
        + "\nTrials:\n"
        + _verbose_trials_detail(runs, outcomes, check, max_chars)
    )


def trial_outcomes_failure_message(
    outcomes: list[tuple[int, TrialFailure | None]],
    target: float,
    label: str,
    *,
    pass_threshold: float = PASS_THRESHOLD,
    max_chars: int = FAILURE_SECTION_MAX_CHARS,
) -> str:
    """Human-readable failure detail for ``trial_outcomes_passed``.

    Pair with ``assert trial_outcomes_passed(outcomes, target),
    trial_outcomes_failure_message(outcomes, target, label)`` in per-skill
    test modules. Renders each failing trial's summary and any structured
    sections without interpreting section labels. ``max_chars`` caps each
    rendered section body (wired to ``--live-eval-failure-max-chars`` by the
    registered tests); zero or negative disables truncation.
    """
    passes = sum(1 for _, err in outcomes if err is None)
    trials = len(outcomes)
    detail = "\n".join(
        _format_trial_failure(idx, err, max_chars)
        for idx, err in outcomes
        if err is not None
    )
    if not trials:
        detail = "  (no gradable trials: every trial errored)"
    return (
        format_posterior_summary(
            label, passes, trials, target, pass_threshold=pass_threshold
        )
        + f" (need p_good > {pass_threshold:.3f}).\nFailing trials:\n"
        + detail
    )


def failing_assertions(
    runs: list[EvalRun],
    assertions: list[dict[str, Any]],
    handlers: dict[str, Callable[[EvalRun], None]],
    target: float,
    *,
    pass_threshold: float = PASS_THRESHOLD,
) -> list[tuple[str, int, int, float]]:
    """For one eval's runs, the assertions whose posterior fails the bar.

    Returns ``(assertion_id, passes, trials, p_good)`` for every assertion
    that `eval_passed` grades as a fail, so a single per-eval report can name
    every assertion that fell short (and pair them with the eval's
    `expected_output`). An empty result means the whole eval cleared the bar.

    Every assertion is expected to have a registered handler;
    `assert_handler_coverage` (run at load time via `load_evals`) guarantees
    this, so a missing handler here is an unvalidated-load bug and surfaces as
    a `KeyError` rather than a silent skip.
    """
    failing: list[tuple[str, int, int, float]] = []
    trials = len(graded_runs(runs))
    for assertion in assertions:
        handler = handlers[assertion["id"]]
        passes = sum(
            1 for _, err in trial_outcomes(runs, handler) if err is None
        )
        if not eval_passed(passes, trials, target, pass_threshold=pass_threshold):
            p_good = posterior_pass_prob(passes, trials, target)
            failing.append((assertion["id"], passes, trials, p_good))
    return failing


def trigger_pass_counts(
    runs: dict[str, list[EvalRun]], evals: list[dict[str, Any]]
) -> list[tuple[str, int, int]]:
    """Per should_trigger eval: (id, trials_invoking_skill, trials_total).

    Counts only graded trials; errored ones carry no trigger evidence.
    """
    return [
        (
            ev["id"],
            sum(r.skill_invoked for r in graded_runs(runs[ev["id"]])),
            len(graded_runs(runs[ev["id"]])),
        )
        for ev in evals
        if ev.get("should_trigger")
    ]
