"""Deciding eval verdicts from repeated trials, the Bayesian way.

Each graded check is a Bernoulli process: on any single `claude -p` run the
skill-with-prompt either satisfies the assertion (with unknown true pass
rate ``theta``) or not. We never observe ``theta`` -- only ``k`` passes out
of ``n`` trials. So instead of thresholding a raw count we put a posterior
on ``theta`` and ask how much of it clears a target rate.

  * Model: ``k ~ Binomial(n, theta)``, prior ``theta ~ Beta(1, 1)`` (uniform).
    Beta is conjugate to the binomial, so the posterior is closed-form:
    ``theta | (k, n) ~ Beta(1 + k, 1 + (n - k))`` -- each batch of trials
    just bumps the two parameters, no sampling.
  * Bar: ``TARGET_RATE`` (default 3/5) is the true pass rate a good skill
    should clear. ``posterior_pass_prob`` returns
    ``p_good = P(theta >= TARGET_RATE | k, n)`` via the regularized
    incomplete beta function (the Beta CDF), stdlib-only.
  * Verdict band: PASS once ``p_good > PASS_THRESHOLD`` (1 - e^-2 ~ 0.865),
    FAIL once ``p_good < FAIL_THRESHOLD`` (e^-2 ~ 0.135); in between the
    evidence is inconclusive and more trials are worth running. The band is
    symmetric so an early unlucky streak does not lock a verdict.

Two concerns live here. First, the adaptive driver: `_eval_checks` derives
the pass/fail checks for an eval, `next_batch_size` decides how many more
trials are worth running given the posterior so far, and `run_eval_adaptive`
loops the two until the verdict is fixed -- capping cost at `MAX_TRIALS`
runs while spending as few as `BATCH_FLOOR` when a clean streak settles it.
Second, the grading rollups (`trial_outcomes`, `eval_passed`,
`trial_outcomes_passed`, `trial_outcomes_failure_message`, `failing_assertions`,
`trigger_pass_counts`) that
per-skill tests use to grade and report on a completed batch of runs.
"""

from __future__ import annotations

import enum
import json
import math
import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from binom_eval.runner import Runner, run_claude_batch
from binom_eval.stream_json import EvalRun

# Beta(1, 1) prior: uniform over theta, i.e. no prior opinion on the rate.
PRIOR_ALPHA = 1.0
PRIOR_BETA = 1.0

# Posterior-mass thresholds for the verdict band, in terms of
# p_good = P(theta >= TARGET_RATE). PASS above the high edge, FAIL below the
# low edge, keep sampling in between. The edges are e^-2 and its complement,
# so the band is symmetric about 1/2 and ~73% wide.
PASS_THRESHOLD = 1.0 - math.exp(-2)  # ~0.8647
FAIL_THRESHOLD = math.exp(-2)  # ~0.1353

# Smallest batch to fire while a verdict is still open. Flooring the
# optimistic shortfall keeps early rounds fanned out for concurrency and --
# because it forces a representative sample before the posterior is allowed
# to commit -- markedly cuts the chance an unlucky streak fails a good skill.
BATCH_FLOOR = 3

# Per-section cap when rendering structured trial failures in pytest output.
# Overridable per run via `--live-eval-failure-max-chars`; zero or negative
# disables truncation.
FAILURE_SECTION_MAX_CHARS = 2000


@dataclass(eq=False)
class AssertionFailure(AssertionError):
    """Structured assertion failure raised by skill assertion handlers.

    ``summary`` is one line for rollups; ``sections`` are skill-defined
    ``(label, body)`` pairs the framework renders without interpreting.
    Every assertion handler should raise this (with or without ``sections``)
    rather than plain ``AssertionError``.
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


class Verdict(enum.Enum):
    """Band verdict for one check, in terms of its posterior mass `p_good`.

    PASS once `p_good` clears `PASS_THRESHOLD`, FAIL once it drops below
    `FAIL_THRESHOLD`, UNDETERMINED in between -- the state in which
    `next_batch_size` keeps running trials.
    """

    PASS = "pass"
    FAIL = "fail"
    UNDETERMINED = "undetermined"


def _betainc(x: float, a: float, b: float) -> float:
    """Regularized incomplete beta ``I_x(a, b)`` -- the CDF of ``Beta(a, b)``.

    Returns ``P(theta <= x)`` for ``theta ~ Beta(a, b)``. Stdlib-only
    (Lentz's continued fraction; ``math.lgamma`` for the front factor), good
    to ~1e-12 over the range this module uses.
    """
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    log_beta = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
    front = math.exp(a * math.log(x) + b * math.log1p(-x) - log_beta)

    def betacf(a: float, b: float, x: float) -> float:
        """Continued fraction for the incomplete beta, via Lentz's algorithm.

        Evaluates the continued fraction that appears in the standard
        ``I_x(a, b)`` expansion (see ``_betainc``), iterating with the
        modified Lentz method: each term updates the running ``c`` and ``d``
        factors, ``guard`` flooring near-zero denominators to ``tiny`` so the
        reciprocals stay finite, until successive terms differ by less than
        ``eps``. The caller multiplies the result by the ``front`` factor and
        divides by ``a`` to recover the regularized value; it is only invoked
        in the region ``x < (a + 1) / (a + b + 2)`` where the fraction
        converges quickly (the reflection in ``_betainc`` handles the rest).

        Args:
          a: First positive shape parameter of the continued fraction.
          b: Second positive shape parameter.
          x: Evaluation point in ``[0, 1]``, within the fast-converging region.

        Returns:
          The value of the continued fraction (not yet scaled by ``front / a``).
        """
        tiny, eps = 1e-30, 1e-14

        def guard(value: float) -> float:
            """Floor near-zero values to ``tiny`` to avoid division by zero."""
            return tiny if abs(value) < tiny else value

        qab, qap, qam = a + b, a + 1.0, a - 1.0
        c = 1.0
        d = 1.0 / guard(1.0 - qab * x / qap)
        h = d
        # Iteration cap: the `eps` convergence test below normally breaks out
        # in well under a few dozen passes over the region this is called in,
        # so this only bounds pathological non-convergence. The exact value is
        # arbitrary (any comfortably-large ceiling works); 377 buys headroom.
        for m in range(1, 377):
            m2 = 2 * m
            aa = m * (b - m) * x / ((qam + m2) * (a + m2))
            d = 1.0 / guard(1.0 + aa * d)
            c = guard(1.0 + aa / c)
            h *= d * c
            aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
            d = 1.0 / guard(1.0 + aa * d)
            c = guard(1.0 + aa / c)
            delta = d * c
            h *= delta
            if abs(delta - 1.0) < eps:
                break
        return h

    # Use the continued fraction in its fast-converging region, else reflect.
    if x < (a + 1.0) / (a + b + 2.0):
        return front * betacf(a, b, x) / a
    return 1.0 - front * betacf(b, a, 1.0 - x) / b


def posterior_pass_prob(passes: int, trials: int, target: float) -> float:
    """``p_good = P(theta >= target)`` under the Beta-binomial posterior.

    With a ``Beta(PRIOR_ALPHA, PRIOR_BETA)`` prior and ``passes`` of
    ``trials`` successes, the posterior is
    ``Beta(PRIOR_ALPHA + passes, PRIOR_BETA + (trials - passes))`` and this
    returns the mass it puts at or above ``target`` (one minus the Beta CDF
    at ``target``). With no trials yet it reduces to the prior's mass above
    ``target``.
    """
    alpha = PRIOR_ALPHA + passes
    beta = PRIOR_BETA + (trials - passes)
    return 1.0 - _betainc(target, alpha, beta)


def _verdict(
    passes: int,
    trials: int,
    target: float,
    *,
    pass_threshold: float = PASS_THRESHOLD,
) -> Verdict:
    """Band verdict for one check.

    PASS once the posterior mass above ``target`` clears ``pass_threshold``,
    FAIL once it drops below ``1 - pass_threshold``, otherwise UNDETERMINED --
    the state in which `next_batch_size` keeps running trials.
    """
    fail_threshold = 1.0 - pass_threshold
    p_good = posterior_pass_prob(passes, trials, target)
    if p_good > pass_threshold:
        return Verdict.PASS
    if p_good < fail_threshold:
        return Verdict.FAIL
    return Verdict.UNDETERMINED


def eval_passed(passes: int, trials: int, target: float) -> bool:
    """Final pass/fail grade for a completed batch of runs.

    The verdict band decides *when to stop*; this decides the *grade* once
    stopping has happened. A PASS-locked run has ``p_good > PASS_THRESHOLD``
    and a FAIL-locked run has ``p_good < FAIL_THRESHOLD``, so grading on
    ``p_good >= 1/2`` agrees with both; the only case it newly resolves is a
    run that exhausted `MAX_TRIALS` still inside the band, which it breaks
    toward whichever side holds the majority of the posterior.
    """
    return posterior_pass_prob(passes, trials, target) >= 0.5


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


def _trigger_check(run: EvalRun) -> None:
    """Assertion-style check that the skill fired (for should_trigger evals)."""
    if not run.skill_invoked:
        raise AssertionFailure(
            "skill was not invoked",
            sections=(("Assistant reply", run.assistant_text or "(empty)"),),
        )


def _no_other_skill_check(skill_name: str) -> Callable[[EvalRun], None]:
    """Return a check that fails when any skill other than ``skill_name`` fires.

    Covers both invocation styles: the ``Skill`` tool (Claude Code) and a
    ``Read`` of ``{skill}/SKILL.md`` (Cursor). When another skill is detected
    the failure message names it so the cause is immediately clear.
    """
    def check(run: EvalRun) -> None:
        other: list[str] = []
        for block in run.tool_uses:
            if block.get("name") == "Skill":
                invoked = block.get("input", {}).get("skill", "")
                if invoked != skill_name:
                    other.append(invoked or str(block.get("input", {})))
            elif block.get("name") == "Read":
                payload = block.get("input", {})
                raw = str(
                    payload.get("path") or payload.get("file_path") or ""
                ).replace("\\", "/")
                parts = raw.split("/")
                if "SKILL.md" in parts and f"/{skill_name}/SKILL.md" not in raw:
                    idx = parts.index("SKILL.md")
                    other.append(parts[idx - 1] if idx > 0 else raw)
        if other:
            raise AssertionFailure(
                f"unexpected skill(s) invoked: {', '.join(other)}",
                sections=(("Tool uses", str(run.tool_uses)),),
            )
    return check


def _eval_checks(
    item: dict[str, Any],
    assertion_handlers: dict[str, Callable[[EvalRun], None]],
    skill_name: str | None = None,
) -> list[Callable[[EvalRun], None]]:
    """The pass/fail checks that decide an eval: its registered assertion
    handlers plus, for should_trigger evals, the skill-invocation check and
    a check that no other skill was used.

    These are exactly the checks whose per-trial outcomes determine whether
    further trials could still change the verdict.
    """
    checks = [
        assertion_handlers[a["id"]]
        for a in item.get("assertions", [])
        if a["id"] in assertion_handlers
    ]
    if item.get("should_trigger"):
        checks.append(_trigger_check)
        if skill_name:
            checks.append(_no_other_skill_check(skill_name))
    return checks


def _check_failures(
    runs: list[EvalRun], check: Callable[[EvalRun], None]
) -> int:
    """Number of `runs` for which `check` fails (raises AssertionFailure)."""
    return sum(1 for _, err in trial_outcomes(runs, check) if err is not None)


def _resolve_shortfall(
    passes: int,
    trials: int,
    target: float,
    remaining: int,
    *,
    pass_threshold: float = PASS_THRESHOLD,
) -> int:
    """Optimistic trials to resolve one undetermined check, capped by budget.

    Looks at the two clean continuations from the current ``(passes,
    trials)`` posterior -- an all-pass streak that would push `p_good` above
    ``pass_threshold``, and an all-fail streak that would push it below
    ``1 - pass_threshold`` -- and returns the shorter. That is the fewest
    trials that *could* settle the check either way. If neither resolves within
    `remaining`, both fall back to `remaining`, so the result is `remaining`.
    """
    fail_threshold = 1.0 - pass_threshold
    to_pass = next(
        (
            i
            for i in range(1, remaining + 1)
            if posterior_pass_prob(passes + i, trials + i, target)
            > pass_threshold
        ),
        remaining,
    )
    to_fail = next(
        (
            i
            for i in range(1, remaining + 1)
            if posterior_pass_prob(passes, trials + i, target)
            < fail_threshold
        ),
        remaining,
    )
    return min(to_pass, to_fail)


def next_batch_size(
    runs: list[EvalRun],
    checks: list[Callable[[EvalRun], None]],
    max_trials: int,
    target: float,
    *,
    pass_threshold: float = PASS_THRESHOLD,
) -> int:
    """How many trials to run next, or 0 once the verdict is fixed.

    The eval fails as soon as *any* check's posterior locks FAIL, and passes
    only once *every* check locks PASS; in between it is undetermined. Given the
    runs so far this returns 0 when the eval is decided (some check FAIL, or
    all checks PASS) or the `max_trials` budget is spent, and otherwise an
    optimistic batch:

      * for each still-undetermined check, `_resolve_shortfall` -- the fewest
        trials that could settle it either way;
      * the eval needs every check settled, so the batch takes the *largest*
        such shortfall;
      * floored at `BATCH_FLOOR` (keeps early rounds fanned out and resists
        unlucky-streak verdicts) and capped by the remaining budget.

    `run_eval_adaptive` re-grades after each batch, so the next one shrinks
    as the posteriors converge. Errored trials count against the budget
    (their cost was spent) but not toward any posterior -- see `graded_runs`.
    """
    remaining = max_trials - len(runs)
    if remaining <= 0:
        return 0
    graded = graded_runs(runs)
    trials_done = len(graded)
    shortfalls: list[int] = []
    for check in checks:
        passes = trials_done - _check_failures(graded, check)
        verdict = _verdict(
            passes, trials_done, target, pass_threshold=pass_threshold
        )
        if verdict is Verdict.FAIL:
            return 0  # eval already fails; no batch can change that.
        if verdict is Verdict.UNDETERMINED:
            shortfalls.append(
                _resolve_shortfall(
                    passes,
                    trials_done,
                    target,
                    remaining,
                    pass_threshold=pass_threshold,
                )
            )
    if not shortfalls:  # every check locked PASS (or there are no checks).
        return 0
    return min(max(max(shortfalls), BATCH_FLOOR), remaining)


def run_eval_adaptive(
    item: dict[str, Any],
    repo_root: Path,
    skill_name: str,
    max_trials: int,
    target: float,
    checks: list[Callable[[EvalRun], None]],
    *,
    pass_threshold: float = PASS_THRESHOLD,
    gate: threading.Semaphore | None = None,
    isolate: bool = False,
    model: str,
    runner: Runner | None = None,
) -> list[EvalRun]:
    """Run trials in optimistic concurrent batches, stopping once the verdict
    is fixed.

    Each round runs `next_batch_size` trials concurrently and re-grades,
    looping until every check's posterior has locked PASS, one has locked
    FAIL, or the `max_trials` budget is spent. This caps cost at `max_trials`
    runs and spends as few as `BATCH_FLOOR` when a clean streak settles every
    check, over however many rounds the outcomes require.

    `gate`, `isolate`, `model`, and `runner` are forwarded to
    `run_claude_batch`: the shared semaphore caps total live calls across this
    eval's batches and any other evals driven in parallel; `isolate` runs
    every trial in its own throwaway copy of `repo_root`; `model` selects the
    specific model used for all trials; `runner` selects the backend (default
    `claude -p`). Batches within one eval still run as sequential rounds (each
    round's verdict decides the next), so concurrency comes from the trials in
    a round plus evals overlapping above this layer.
    """
    runs: list[EvalRun] = []
    batch = next_batch_size(
        runs, checks, max_trials, target, pass_threshold=pass_threshold
    )
    while batch > 0:
        runs.extend(
            run_claude_batch(
                item,
                repo_root,
                skill_name,
                batch,
                gate=gate,
                isolate=isolate,
                model=model,
                runner=runner,
            )
        )
        batch = next_batch_size(
            runs, checks, max_trials, target, pass_threshold=pass_threshold
        )
    return runs


def assert_handler_coverage(
    evals: list[dict[str, Any]],
    handlers: dict[str, Callable[[EvalRun], None]],
) -> None:
    """Verify every assertion across `evals` has a registered handler.

    Raises:
        KeyError: naming every ``eval_id::assertion_id`` whose `id` is not in
            `handlers`. An assertion with no handler can never be graded, so
            this is a misconfigured suite -- failing once, eagerly, with the
            full list beats discovering each gap later during grading.
    """
    missing = [
        f"{ev['id']}::{assertion['id']}"
        for ev in evals
        for assertion in ev.get("assertions", [])
        if assertion["id"] not in handlers
    ]
    if missing:
        raise KeyError(
            "no handler registered for assertion(s): "
            + ", ".join(missing)
            + "; add them to ASSERTION_HANDLERS"
        )


def expand_eval_item(item: dict[str, Any], eval_dir: Path) -> dict[str, Any]:
    """Return one eval dict with ``prompt_template`` + ``fixture`` expanded.

    For ``should_trigger`` evals the prompt is extended with a constraint
    naming the only skill that should be used. The skill name is derived from
    the directory layout: ``evals.json`` sits two levels below the skill root
    (``{skill}/evals/{lang}/evals.json``), so ``eval_dir.parents[1].name``
    is the skill name without needing a redundant field in ``evals.json``.
    """
    expanded = dict(item)
    fixture = expanded.pop("fixture", None)
    template = expanded.pop("prompt_template", None)
    if fixture is not None and template is not None:
        content = (eval_dir / fixture).read_text(encoding="utf-8")
        if expanded.get("should_trigger"):
            skill_name = eval_dir.parents[1].name
            template += (
                f"\n\nUse only the `{skill_name}` skill."
                " Do not invoke any other skill."
            )
        expanded["prompt"] = template.format(fixture=content)
        expanded["prompt_input"] = content
    elif fixture is not None or template is not None:
        raise ValueError(
            f"eval {expanded.get('id')!r} needs both prompt_template and fixture"
        )
    return expanded


def expand_evals(evals_path: Path) -> list[dict[str, Any]]:
    """Load ``evals.json`` and expand any ``prompt_template`` + ``fixture`` pairs."""
    eval_dir = evals_path.parent
    raw = json.loads(evals_path.read_text(encoding="utf-8"))["evals"]
    return [expand_eval_item(item, eval_dir) for item in raw]


def load_evals(
    evals_path: Path,
    handlers: dict[str, Callable[[EvalRun], None]] | None = None,
) -> list[dict[str, Any]]:
    """Read an `evals.json` file and return its list of eval items.

    Eval entries may supply a finished ``prompt`` or a ``prompt_template`` plus
    ``fixture`` path (relative to the directory containing ``evals.json``).

    Args:
        evals_path: Path to a skill's `evals.json`, an object with an
            `"evals"` key.
        handlers: when given, every assertion id across the loaded evals must
            have a registered handler; otherwise a single `KeyError` names all
            the gaps. This catches a misconfigured suite at load time so
            downstream grading never meets an ungradeable assertion.

    Returns:
        The value of the file's top-level `"evals"` array, with prompts expanded.
    """
    evals = expand_evals(evals_path)
    if handlers is not None:
        assert_handler_coverage(evals, handlers)
    return evals


def _capture_trial_failure(exc: AssertionFailure) -> TrialFailure:
    return TrialFailure(exc.summary, exc.sections)


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


def trial_outcomes(
    runs: list[EvalRun], check: Callable[[EvalRun], None]
) -> list[tuple[int, TrialFailure | None]]:
    """Run `check` against each gradeable trial run, capturing its result.

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
    outcomes: list[tuple[int, str | None]], target: float
) -> bool:
    """True when trial outcomes clear the posterior bar at ``target``."""
    passes = sum(1 for _, err in outcomes if err is None)
    return eval_passed(passes, len(outcomes), target)


def format_posterior_summary(
    label: str,
    passes: int,
    trials: int,
    target: float,
) -> str:
    """One-line summary of ``P(rate >= p0 | k, n)`` for pytest output.

    ``p0`` is the target pass rate, ``k`` the pass count, ``n`` the trial
    count. Returns ``p_good = P(rate >= p0 | k, n)`` under the Beta(1, 1)
    posterior.
    """
    p_good = posterior_pass_prob(passes, trials, target)
    return (
        f"{label}: {passes}/{trials} trials passed; "
        f"P(rate >= {target:.3f} | k={passes}, n={trials}) = {p_good:.3f}"
    )


def trial_outcomes_posterior_summary(
    outcomes: list[tuple[int, TrialFailure | None]],
    target: float,
    label: str,
) -> str:
    """``format_posterior_summary`` for a ``trial_outcomes`` result list."""
    passes = sum(1 for _, err in outcomes if err is None)
    return format_posterior_summary(label, passes, len(outcomes), target)


def trial_outcomes_failure_message(
    outcomes: list[tuple[int, TrialFailure | None]],
    target: float,
    label: str,
    *,
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
        detail = "  (no gradeable trials: every trial errored)"
    return (
        format_posterior_summary(label, passes, trials, target)
        + " (need >= 0.5).\nFailing trials:\n"
        + detail
    )


def failing_assertions(
    runs: list[EvalRun],
    assertions: list[dict[str, Any]],
    handlers: dict[str, Callable[[EvalRun], None]],
    target: float,
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
        if not eval_passed(passes, trials, target):
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