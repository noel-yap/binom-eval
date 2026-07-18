"""Deciding eval verdicts from repeated trials, the Bayesian way.

Each graded check is a Bernoulli process: on any single `claude -p` run the
skill-with-prompt either satisfies the assertion (with unknown true pass
rate ``θ``) or not. We never observe ``θ`` -- only ``k`` passes out
of ``n`` trials. So instead of thresholding a raw count we put a posterior
on ``θ`` and ask how much of it clears a target rate.

  * Model: ``k ~ Binomial(n, θ)``, prior ``θ ~ Beta(1, 1)`` (uniform).
    Beta is conjugate to the binomial, so the posterior is closed-form:
    ``Θ | (k, n) ~ Beta(1 + k, 1 + (n - k))`` -- each batch of trials
    just bumps the two parameters, no sampling.
  * Bar: ``TARGET_RATE`` (default 3/5) is the true pass rate a good skill
    should clear. ``posterior_pass_prob`` returns
    ``p_good = P(θ ≥ TARGET_RATE | k, n)`` via the regularized
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

from binom_eval.runner import Runner, run_eval_batch
from binom_eval.posterior import (
    FAIL_THRESHOLD, PASS_THRESHOLD, PRIOR_ALPHA, PRIOR_BETA, Verdict,
    _betainc, _verdict, eval_passed, max_target_at_pass_threshold,
    posterior_pass_prob,
)
from binom_eval.assertions import (
    AssertionFailure, TrialFailure, _assertion_sections, _capture_trial_failure,
    _default_pass_sections, assert_check, evaluate_check,
)
from binom_eval.loading import (
    assert_handler_coverage, expand_eval_item, expand_evals, load_evals,
)
from binom_eval.reporting import (
    FAILURE_SECTION_MAX_CHARS, _format_trial_failure, _truncate_section_body,
    _verbose_trials_detail, failing_assertions, format_posterior_summary,
    graded_runs, graded_runs_verbose_message, trial_outcomes,
    trial_outcomes_failure_message, trial_outcomes_passed,
    trial_outcomes_posterior_summary, trial_outcomes_verbose_message,
    trigger_pass_counts,
)
from binom_eval.driver import (
    BATCH_FLOOR, _check_failures, _eval_checks, _no_other_skill_check,
    _resolve_shortfall, _trigger_check, next_batch_size, run_eval_adaptive,
)
from binom_eval.progress import (
    ProgressEvent, ProgressRenderer, PlainRenderer, TtyRenderer, make_renderer,
)
