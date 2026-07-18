from __future__ import annotations

import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from binom_eval.runner import DEFAULT_TIMEOUT_SECONDS, Runner, run_eval_batch
from binom_eval.stream_json import EvalRun
from binom_eval.posterior import PASS_THRESHOLD, Verdict, _verdict, posterior_pass_prob
from binom_eval.assertions import assert_check
from binom_eval.reporting import graded_runs, trial_outcomes
from binom_eval.progress import ProgressEvent, ProgressRenderer

# Smallest batch to fire while a verdict is still open. Flooring the
# optimistic shortfall keeps early rounds fanned out for concurrency and --
# because it forces a representative sample before the posterior is allowed
# to commit -- markedly cuts the chance an unlucky streak fails a good skill.
BATCH_FLOOR = 3


def _trigger_check(run: EvalRun) -> None:
    """Assertion-style check that the skill fired (for should_trigger evals)."""
    sections = (("Assistant reply", run.assistant_text or "(empty)"),)
    assert_check(run.skill_invoked, "skill was not invoked", sections=sections)


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
        if run.tool_uses:
            sections = (("Tool uses", str(run.tool_uses)),)
        else:
            sections = (("Assistant reply", run.assistant_text or "(empty)"),)
        assert_check(
            not other,
            f"unexpected skill(s) invoked: {', '.join(other)}",
            sections=sections,
        )
    return check


def _eval_verdict(
    runs: list[EvalRun],
    checks: list[Callable[[EvalRun], None]],
    target: float,
    *,
    pass_threshold: float = PASS_THRESHOLD,
) -> Verdict:
    """Compute the overall eval verdict given the runs and checks so far.

    Returns FAIL if any check has locked to FAIL; PASS if all checks have
    locked to PASS; UNDETERMINED otherwise.
    """
    graded = graded_runs(runs)
    trials_done = len(graded)
    if trials_done == 0:
        return Verdict.UNDETERMINED
    all_pass = True
    for check in checks:
        passes = trials_done - _check_failures(graded, check)
        v = _verdict(passes, trials_done, target, pass_threshold=pass_threshold)
        if v is Verdict.FAIL:
            return Verdict.FAIL
        if v is Verdict.UNDETERMINED:
            all_pass = False
    return Verdict.PASS if all_pass else Verdict.UNDETERMINED


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
    min_trials: int = 0,
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

    ``min_trials`` only postpones stopping: when the adaptive logic would
    stop (verdict locked) before ``min_trials`` trials have run, the driver
    keeps running until the floor is met. It never changes the size of
    batches the adaptive logic already wants.
    """
    remaining = max_trials - len(runs)
    if remaining <= 0:
        return 0
    # A "stop" (batch 0) is postponed while the min-trials floor is unmet:
    # instead of stopping, run just enough trials to reach the floor.
    if len(runs) < min_trials:
        stop_batch = min(min_trials - len(runs), remaining)
    else:
        stop_batch = 0
    graded = graded_runs(runs)
    trials_done = len(graded)
    shortfalls: list[int] = []
    for check in checks:
        passes = trials_done - _check_failures(graded, check)
        verdict = _verdict(
            passes, trials_done, target, pass_threshold=pass_threshold
        )
        if verdict is Verdict.FAIL:
            # Eval already fails; no batch can change that.
            return stop_batch
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
        return stop_batch
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
    min_trials: int = 0,
    gate: threading.Semaphore | None = None,
    isolate: bool = False,
    model: str,
    runner: Runner,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
    on_progress: ProgressRenderer | None = None,
    batch_runner: Callable[..., list[EvalRun]] = run_eval_batch,
) -> list[EvalRun]:
    """Run trials in optimistic concurrent batches, stopping once the verdict
    is fixed.

    Each round runs `next_batch_size` trials concurrently and re-grades,
    looping until every check's posterior has locked PASS, one has locked
    FAIL, or the `max_trials` budget is spent. This caps cost at `max_trials`
    runs and spends as few as `BATCH_FLOOR` when a clean streak settles every
    check, over however many rounds the outcomes require.

    `gate`, `isolate`, `model`, `runner`, and `timeout` are forwarded to
    `run_eval_batch`: the shared semaphore caps total live calls across this
    eval's batches and any other evals driven in parallel; `isolate` runs
    every trial in its own throwaway copy of `repo_root`; `model` selects the
    specific model used for all trials; `runner` is the backend every trial
    runs against (backend-agnostic -- `ClaudeRunner`, `CursorRunner`, ...).
    `timeout` sets the per-trial subprocess deadline in seconds.
    Batches within one eval still run as sequential rounds (each round's
    verdict decides the next), so concurrency comes from the trials in a round
    plus evals overlapping above this layer. `on_progress` is an optional
    progress renderer to report per-batch and completion events. `batch_runner`
    defaults to `run_eval_batch` and exists so callers/tests can inject a
    different batch executor.
    """
    runs: list[EvalRun] = []
    batch_num = 0
    eval_start = time.monotonic()
    batch = next_batch_size(
        runs,
        checks,
        max_trials,
        target,
        pass_threshold=pass_threshold,
        min_trials=min_trials,
    )
    while batch > 0:
        batch_num += 1
        batch_start = time.monotonic()
        runs.extend(
            batch_runner(
                item,
                repo_root,
                skill_name,
                batch,
                gate=gate,
                isolate=isolate,
                model=model,
                runner=runner,
                timeout=timeout,
            )
        )
        batch_elapsed = time.monotonic() - batch_start
        total_elapsed = time.monotonic() - eval_start
        if on_progress is not None:
            on_progress.render(
                ProgressEvent(
                    kind="batch",
                    eval_id=item["id"],
                    batch_num=batch_num,
                    batch_size=batch,
                    trials_run=len(runs),
                    trials_max=max_trials,
                    batch_elapsed=batch_elapsed,
                    total_elapsed=total_elapsed,
                    verdict=_eval_verdict(
                        runs, checks, target, pass_threshold=pass_threshold
                    ),
                )
            )
        batch = next_batch_size(
            runs,
            checks,
            max_trials,
            target,
            pass_threshold=pass_threshold,
            min_trials=min_trials,
        )
    if on_progress is not None:
        on_progress.render(
            ProgressEvent(
                kind="eval_done",
                eval_id=item["id"],
                batch_num=0,
                batch_size=0,
                trials_run=len(runs),
                trials_max=max_trials,
                batch_elapsed=0.0,
                total_elapsed=time.monotonic() - eval_start,
                verdict=_eval_verdict(
                    runs, checks, target, pass_threshold=pass_threshold
                ),
            )
        )
    return runs
