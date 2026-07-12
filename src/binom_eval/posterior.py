from __future__ import annotations

import enum
import math

# Beta(1, 1) prior: uniform over θ, i.e. no prior opinion on the rate.
PRIOR_ALPHA = 1.0
PRIOR_BETA = 1.0

# Posterior-mass thresholds for the verdict band, in terms of
# p_good = P(θ ≥ TARGET_RATE). PASS above the high edge, FAIL below the
# low edge, keep sampling in between. The edges are e^-2 and its complement,
# so the band is symmetric about 1/2 and ~73% wide.
PASS_THRESHOLD = 1.0 - math.exp(-2)  # ~0.8647
FAIL_THRESHOLD = math.exp(-2)  # ~0.1353


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

    Returns ``P(θ ≤ x)`` for ``θ ~ Beta(a, b)``. Stdlib-only
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
    """``p_good = P(θ ≥ target)`` under the Beta-binomial posterior.

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


def max_target_at_pass_threshold(
    passes: int,
    trials: int,
    pass_threshold: float,
) -> float:
    """Highest target rate at which the posterior still PASS-locks.

    Returns the largest ``θ₀`` in ``[0, 1]`` with
    ``P(θ ≥ θ₀ | k, n) > τ`` (matching ``_verdict``'s strict PASS edge at
    ``τ = pass_threshold``). Returns ``0.0`` when no target clears ``τ``.
    """
    if pass_threshold >= 1.0:
        return 0.0
    lo, hi = 0.0, 1.0
    for _ in range(64):
        mid = (lo + hi) / 2.0
        if posterior_pass_prob(passes, trials, mid) > pass_threshold:
            lo = mid
        else:
            hi = mid
    return lo


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
