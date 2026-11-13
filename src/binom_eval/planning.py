"""Trial-budget planning: derive min/max trials from θ₀, τ and δ.

The verdict rule (PASS iff ``p_good > τ``) is unchanged; this module only
answers "how many trials does that rule need?":

  * ``min_trials`` -- the fewest trials at which an all-pass streak can clear
    τ at all (closed form in θ₀ and τ).
  * ``k_min`` -- the fewest passes out of ``N`` trials that still PASS.
  * ``max_trials`` -- the smallest ``N`` at which a skill at ``θ₀ + δ`` PASSes
    with probability at least τ while one at ``θ₀ - δ`` PASSes with
    probability at most ``1 - τ`` (a symmetric indifference zone ``θ₀ ± δ``).
  * ``adaptive_power`` -- the exact probability the *adaptive* driver ends in
    PASS within a budget, which is lower than the fixed-``N`` figure because
    the driver can lock FAIL early on an unlucky start.
  * ``resolve_trial_bounds`` -- combines the derived values with explicit
    ``--live-eval-min-trials`` / ``--live-eval-max-trials`` per the override
    rules (explicit flags always win).

Stdlib only.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from binom_eval.posterior import (
    PASS_THRESHOLD,
    Verdict,
    _verdict,
    posterior_pass_prob,
)

# Largest N the search for `max_trials` will try before giving up.
MAX_TRIALS_CAP = 1000


def validate_planning_inputs(
    target: float, pass_threshold: float, delta: float
) -> None:
    """Raise ``ValueError`` unless ``δ > 0``, ``θ₀ ± δ`` lies strictly inside
    ``(0, 1)`` and ``τ`` lies strictly inside ``(0.5, 1)``.
    """
    if not 0.5 < pass_threshold < 1.0:
        raise ValueError(
            f"pass threshold must be strictly between 0.5 and 1.0, "
            f"got {pass_threshold}"
        )
    if delta <= 0.0:
        raise ValueError(f"delta must be > 0, got {delta}")
    if target - delta <= 0.0:
        raise ValueError(
            f"target rate - delta must be > 0, got {target} - {delta}"
        )
    if target + delta >= 1.0:
        raise ValueError(
            f"target rate + delta must be < 1, got {target} + {delta}"
        )


def min_trials(target: float, pass_threshold: float = PASS_THRESHOLD) -> int:
    """Fewest trials at which ``n`` of ``n`` passes can clear ``τ``.

    Closed form ``⌈ln(1-τ) / ln θ₀⌉ - 1`` (from ``1 - θ₀^(n+1) > τ``), then
    nudged against the actual posterior so float rounding at an exact
    boundary cannot disagree with the verdict.
    """
    if not 0.0 < target < 1.0:
        raise ValueError(f"target rate must be in (0, 1), got {target}")
    n = max(
        0,
        math.ceil(math.log(1.0 - pass_threshold) / math.log(target)) - 1,
    )
    while n > 0 and posterior_pass_prob(n - 1, n - 1, target) > pass_threshold:
        n -= 1
    while not posterior_pass_prob(n, n, target) > pass_threshold:
        n += 1
    return n


def k_min(
    trials: int, target: float, pass_threshold: float = PASS_THRESHOLD
) -> int | None:
    """Fewest passes out of ``trials`` that still PASS, or ``None`` if even
    ``trials`` of ``trials`` cannot clear ``τ``.
    """
    if not posterior_pass_prob(trials, trials, target) > pass_threshold:
        return None
    lo, hi = 0, trials  # `hi` passes; `p_good` is increasing in k.
    while lo < hi:
        mid = (lo + hi) // 2
        if posterior_pass_prob(mid, trials, target) > pass_threshold:
            hi = mid
        else:
            lo = mid + 1
    return lo


def _binom_sf(k: int, trials: int, p: float) -> float:
    """``P(Binomial(trials, p) >= k)``, summed in log space for stability."""
    if k <= 0:
        return 1.0
    if k > trials:
        return 0.0
    log_p, log_q = math.log(p), math.log1p(-p)
    total = 0.0
    for j in range(k, trials + 1):
        log_term = (
            math.lgamma(trials + 1)
            - math.lgamma(j + 1)
            - math.lgamma(trials - j + 1)
            + j * log_p
            + (trials - j) * log_q
        )
        total += math.exp(log_term)
    return min(1.0, total)


def max_trials(
    target: float,
    delta: float,
    pass_threshold: float = PASS_THRESHOLD,
    *,
    batch_max: int = 1,
) -> int:
    """Smallest ``N`` meeting both indifference-zone conditions.

    With ``k_min(N)`` the fewest passes that PASS at ``N`` trials:
    ``P(Bin(N, θ₀+δ) >= k_min) >= τ`` and ``P(Bin(N, θ₀-δ) >= k_min) <= 1-τ``.
    Scans upward from ``min_trials`` and rounds the result up to a multiple
    of ``batch_max``. Raises ``ValueError`` if no ``N`` up to
    ``MAX_TRIALS_CAP`` qualifies.
    """
    validate_planning_inputs(target, pass_threshold, delta)
    for n in range(min_trials(target, pass_threshold), MAX_TRIALS_CAP + 1):
        k = k_min(n, target, pass_threshold)
        if k is None:
            continue
        if (
            _binom_sf(k, n, target + delta) >= pass_threshold
            and _binom_sf(k, n, target - delta) <= 1.0 - pass_threshold
        ):
            step = max(1, batch_max)
            return -(-n // step) * step
    raise ValueError(
        f"no trial count up to {MAX_TRIALS_CAP} separates θ₀+δ from θ₀-δ "
        f"at target={target}, delta={delta}, pass_threshold="
        f"{pass_threshold}; increase delta or lower the pass threshold"
    )


def implied_delta(
    trials: int, target: float, pass_threshold: float = PASS_THRESHOLD
) -> float | None:
    """Smallest ``δ`` that ``trials`` fixed trials resolve: the inverse of
    ``max_trials``.

    With ``k = k_min(trials)``, the smallest ``δ`` such that
    ``P(Bin(n, θ₀+δ) >= k) >= τ`` and ``P(Bin(n, θ₀-δ) >= k) <= 1-τ``
    (both are monotone in ``δ``, so bisection finds the boundary). Returns
    ``None`` if ``trials`` is too few to PASS at all, or if no ``δ`` with
    ``θ₀ ± δ`` inside ``(0, 1)`` satisfies both conditions.
    """
    k = k_min(trials, target, pass_threshold)
    if k is None:
        return None

    def resolves(delta: float) -> bool:
        return (
            _binom_sf(k, trials, target + delta) >= pass_threshold
            and _binom_sf(k, trials, target - delta) <= 1.0 - pass_threshold
        )

    hi = min(target, 1.0 - target) * (1.0 - 1e-9)
    if not resolves(hi):
        return None
    lo = 0.0
    for _ in range(60):
        mid = (lo + hi) / 2.0
        if resolves(mid):
            hi = mid
        else:
            lo = mid
    return hi


def adaptive_outcome_masses(
    max_trials: int,
    theta: float,
    target: float,
    pass_threshold: float = PASS_THRESHOLD,
    *,
    min_trials: int = 0,
) -> tuple[float, float, float]:
    """Exact ``(P(PASS), P(FAIL), P(exhausted undetermined))`` for a skill at
    true rate ``theta`` under one-trial-at-a-time adaptive stopping.

    A state ``(n, k)`` absorbs once ``_verdict`` locks PASS or FAIL and
    ``n >= min_trials``; a path still undetermined at ``max_trials`` grades
    FAIL (see ``eval_passed``) but is reported separately. The three masses
    sum to 1.
    """
    passed = failed = 0.0
    live = {0: 1.0}  # k -> probability mass, all at the current n
    for n in range(max_trials + 1):
        nxt: dict[int, float] = {}
        for k, mass in live.items():
            verdict = _verdict(k, n, target, pass_threshold=pass_threshold)
            if n >= min_trials and verdict is Verdict.PASS:
                passed += mass
            elif n >= min_trials and verdict is Verdict.FAIL:
                failed += mass
            elif n == max_trials:
                continue  # exhausted; accounted for below
            else:
                nxt[k + 1] = nxt.get(k + 1, 0.0) + mass * theta
                nxt[k] = nxt.get(k, 0.0) + mass * (1.0 - theta)
        live = nxt
    exhausted = max(0.0, 1.0 - passed - failed)
    return passed, failed, exhausted


def adaptive_power(
    max_trials: int,
    theta: float,
    target: float,
    pass_threshold: float = PASS_THRESHOLD,
    *,
    min_trials: int = 0,
) -> float:
    """Probability the adaptive driver ends in PASS within ``max_trials``
    for a skill whose true pass rate is ``theta``.
    """
    return adaptive_outcome_masses(
        max_trials, theta, target, pass_threshold, min_trials=min_trials
    )[0]


@dataclass
class TrialPlan:
    """Effective trial bounds plus the derived values behind them."""

    min_trials: int
    max_trials: int
    derived_min: int | None = None
    derived_max: int | None = None
    power: float | None = None
    notes: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def summary(self) -> str:
        """One-line description for the run header."""
        parts = [f"trials min={self.min_trials} max={self.max_trials}"]
        if self.derived_min is not None:
            parts.append(f"derived N_min={self.derived_min}")
        if self.derived_max is not None:
            parts.append(f"derived N_max={self.derived_max}")
        if self.power is not None:
            parts.append(f"adaptive power at N_max={self.power:.3f}")
        return "; ".join(parts)


def resolve_trial_bounds(
    target: float,
    pass_threshold: float,
    batch_max: int,
    *,
    delta: float,
    min_explicit: int | None,
    max_explicit: int | None,
) -> TrialPlan:
    """Effective min/max trials from derived values and explicit flags.

    Explicit flags always win. With neither given, the values derived from
    ``θ₀``, ``τ`` and ``delta`` are used. ``max_explicit < N_min`` and
    ``min > max`` raise ``ValueError``; an explicit max whose adaptive power
    at ``θ₀+δ`` falls below ``τ`` only adds a warning.
    """
    validate_planning_inputs(target, pass_threshold, delta)
    derived_min = min_trials(target, pass_threshold)
    derived_max = max_trials(target, delta, pass_threshold, batch_max=batch_max)
    plan = TrialPlan(
        derived_min if min_explicit is None else min_explicit,
        derived_max if max_explicit is None else max_explicit,
        derived_min=derived_min,
        derived_max=derived_max,
    )

    if min_explicit is not None and min_explicit < derived_min:
        plan.notes.append(
            f"--live-eval-min-trials ({min_explicit}) is below the "
            f"derived N_min ({derived_min}); no effect on PASS"
        )
    if max_explicit is not None and max_explicit < derived_min:
        raise ValueError(
            f"--live-eval-max-trials ({max_explicit}) is below the "
            f"derived N_min ({derived_min}); PASS is impossible"
        )
    if plan.min_trials > plan.max_trials:
        raise ValueError(
            f"--live-eval-min-trials ({plan.min_trials}) must not exceed "
            f"--live-eval-max-trials ({plan.max_trials})"
        )

    plan.power = adaptive_power(
        plan.max_trials,
        target + delta,
        target,
        pass_threshold,
        min_trials=plan.min_trials,
    )
    if max_explicit is not None and plan.power < pass_threshold:
        plan.warnings.append(
            f"--live-eval-max-trials ({max_explicit}) is underpowered: "
            f"a skill at θ₀+δ={target + delta:.3f} PASSes with adaptive "
            f"power {plan.power:.3f} < τ={pass_threshold:.3f} "
            f"(derived N_max={derived_max})"
        )
    return plan
