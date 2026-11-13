"""Unit tests for `binom_eval.planning` (derived min/max trials).

Pins `min_trials`, `k_min`, `max_trials`, `adaptive_power` and
`resolve_trial_bounds` against brute-force computations from the posterior
and the examples worked out for the design.
"""

from __future__ import annotations

import pytest

from binom_eval import (
    PASS_THRESHOLD,
    adaptive_outcome_masses,
    adaptive_power,
    implied_delta,
    k_min,
    max_trials,
    min_trials,
    posterior_pass_prob,
    resolve_trial_bounds,
)
from binom_eval.planning import (
    MAX_TRIALS_CAP,
    _binom_sf,
    validate_planning_inputs,
)

TAU = PASS_THRESHOLD
GRID = [
    (0.6, TAU), (0.6, 0.9), (0.7, 0.95), (0.8, 0.9), (0.8, 0.95),
    (0.9, 0.9), (0.5, 0.6), (0.99, 0.75),
]


def _smallest_passing_k(n: int, target: float, tau: float) -> int | None:
    """Smallest k whose posterior clears ``tau`` at ``n`` trials, else None."""
    return next(
        (k for k in range(n + 1) if posterior_pass_prob(k, n, target) > tau),
        None,
    )


def _first_all_pass_streak(target: float, tau: float) -> int:
    """Smallest n at which n passes out of n trials clears ``tau``."""
    return next(
        n for n in range(2000) if posterior_pass_prob(n, n, target) > tau
    )


def _brute_force_ok(
    n: int, target: float, delta: float, tau: float
) -> bool:
    """Both power conditions hold at ``n`` for the smallest passing k."""
    k = _smallest_passing_k(n, target, tau)
    return (
        k is not None
        and _binom_sf(k, n, target + delta) >= tau
        and _binom_sf(k, n, target - delta) <= 1 - tau
    )


def _brute_force_max_trials(target: float, delta: float, tau: float) -> int:
    return next(
        n for n in range(1, 500) if _brute_force_ok(n, target, delta, tau)
    )


class TestBruteForceHelpers:
    def test_smallest_passing_k_none_when_n_too_small(self) -> None:
        assert _smallest_passing_k(1, 0.9, 0.95) is None

    def test_smallest_passing_k_found(self) -> None:
        k = _smallest_passing_k(10, 0.6, 0.9)
        assert posterior_pass_prob(k, 10, 0.6) > 0.9
        assert posterior_pass_prob(k - 1, 10, 0.6) <= 0.9

    def test_first_all_pass_streak_known_value(self) -> None:
        assert _first_all_pass_streak(0.6, TAU) == 3

    def test_ok_false_when_no_passing_k(self) -> None:
        assert not _brute_force_ok(1, 0.9, 0.05, 0.95)

    def test_ok_false_when_upper_condition_fails(self) -> None:
        n = max_trials(0.6, 0.15, TAU)
        assert not _brute_force_ok(n, 0.6, 0.15, 0.999999)

    def test_ok_false_when_lower_condition_fails(self) -> None:
        # Tiny delta: upper tail is below tau and lower tail is above 1 - tau.
        assert not _brute_force_ok(5, 0.6, 0.0001, TAU)

    def test_ok_true_when_both_hold(self) -> None:
        n = max_trials(0.6, 0.2, TAU)
        assert _brute_force_ok(n, 0.6, 0.2, TAU)

    def test_max_trials_is_first_ok(self) -> None:
        n = _brute_force_max_trials(0.6, 0.2, TAU)
        assert _brute_force_ok(n, 0.6, 0.2, TAU)
        assert not _brute_force_ok(n - 1, 0.6, 0.2, TAU)


class TestMinTrials:
    @pytest.mark.parametrize(("target", "tau"), GRID)
    def test_matches_brute_force_first_all_pass_streak(
        self, target: float, tau: float
    ) -> None:
        assert min_trials(target, tau) == _first_all_pass_streak(target, tau)

    @pytest.mark.parametrize(
        ("target", "tau", "expected"),
        [(0.6, TAU, 3), (0.8, 0.9, 10), (0.8, 0.95, 13)],
    )
    def test_examples(self, target: float, tau: float, expected: int) -> None:
        assert min_trials(target, tau) == expected

    def test_exact_boundary_is_strict(self) -> None:
        # θ₀=0.5, τ=0.75: 1 - 0.5^(n+1) > 0.75 first holds at n=2 (0.875);
        # n=1 gives exactly 0.75, which is not > τ.
        assert min_trials(0.5, 0.75) == 2

    def test_rejects_target_outside_unit_interval(self) -> None:
        with pytest.raises(ValueError, match="target rate"):
            min_trials(1.0, 0.9)


class TestKMin:
    def test_none_below_min_trials(self) -> None:
        assert k_min(min_trials(0.6, TAU) - 1, 0.6, TAU) is None

    def test_equals_trials_at_min_trials(self) -> None:
        n = min_trials(0.6, TAU)
        assert k_min(n, 0.6, TAU) == n

    @pytest.mark.parametrize("trials", [5, 12, 22, 60])
    def test_is_smallest_passing_count(self, trials: int) -> None:
        k = k_min(trials, 0.6, TAU)
        assert k is not None
        assert posterior_pass_prob(k, trials, 0.6) > TAU
        assert not posterior_pass_prob(k - 1, trials, 0.6) > TAU


class TestMaxTrials:
    @pytest.mark.parametrize(
        ("target", "delta", "tau", "expected"),
        [
            (0.6, 0.2, TAU, 22),
            (0.6, 0.1, TAU, 107),
            (0.8, 0.1, 0.9, 85),
            (0.9, 0.05, 0.9, 174),
        ],
    )
    def test_examples(
        self, target: float, delta: float, tau: float, expected: int
    ) -> None:
        assert max_trials(target, delta, tau) == expected

    @pytest.mark.parametrize(("target", "delta", "tau"), [
        (0.6, 0.2, TAU), (0.7, 0.15, 0.9), (0.5, 0.25, 0.8),
    ])
    def test_matches_brute_force(
        self, target: float, delta: float, tau: float
    ) -> None:
        assert max_trials(target, delta, tau) == _brute_force_max_trials(
            target, delta, tau
        )

    def test_both_conditions_hold_at_result(self) -> None:
        n = max_trials(0.6, 0.15, TAU)
        k = k_min(n, 0.6, TAU)
        assert k is not None
        assert _binom_sf(k, n, 0.75) >= TAU
        assert _binom_sf(k, n, 0.45) <= 1 - TAU

    def test_monotone_in_delta(self) -> None:
        deltas = [0.05, 0.08, 0.1, 0.15, 0.2, 0.3]
        values = [max_trials(0.6, d, TAU) for d in deltas]
        assert values == sorted(values, reverse=True)

    def test_rounds_up_to_batch_multiple(self) -> None:
        assert max_trials(0.6, 0.2, TAU, batch_max=3) == 24
        assert max_trials(0.6, 0.2, TAU, batch_max=4) == 24
        assert max_trials(0.6, 0.2, TAU, batch_max=22) == 22

    def test_unreachable_cap_raises(self) -> None:
        with pytest.raises(ValueError, match=str(MAX_TRIALS_CAP)):
            max_trials(0.6, 0.001, TAU)

    @pytest.mark.parametrize(("target", "delta", "tau", "match"), [
        (0.6, 0.0, TAU, "delta must be > 0"),
        (0.6, -0.1, TAU, "delta must be > 0"),
        (0.1, 0.1, TAU, "target rate - delta"),
        (0.9, 0.1, TAU, "target rate \\+ delta"),
        (0.6, 0.1, 0.5, "strictly between"),
        (0.6, 0.1, 1.0, "strictly between"),
    ])
    def test_validation(
        self, target: float, delta: float, tau: float, match: str
    ) -> None:
        with pytest.raises(ValueError, match=match):
            validate_planning_inputs(target, tau, delta)
        with pytest.raises(ValueError, match=match):
            max_trials(target, delta, tau)


class TestImpliedDelta:
    def test_63_trials_at_default_target(self):
        assert implied_delta(63, 0.6, TAU) == pytest.approx(0.1360, abs=1e-3)

    def test_none_when_too_few_trials_to_pass(self):
        assert implied_delta(min_trials(0.6, TAU) - 1, 0.6, TAU) is None

    def test_none_when_no_delta_inside_unit_interval_resolves(self):
        assert implied_delta(3, 0.99, TAU) is None

    def test_is_boundary_of_both_indifference_conditions(self):
        k = k_min(63, 0.6, TAU)

        def resolves(delta: float) -> bool:
            return (
                _binom_sf(k, 63, 0.6 + delta) >= TAU
                and _binom_sf(k, 63, 0.6 - delta) <= 1.0 - TAU
            )

        delta = implied_delta(63, 0.6, TAU)
        assert resolves(delta + 1e-9)
        assert not resolves(delta - 1e-6)

    def test_shrinks_as_trials_grow(self):
        assert implied_delta(69, 0.6, TAU) < implied_delta(20, 0.6, TAU)


class TestAdaptivePower:
    @pytest.mark.parametrize("theta", [0.3, 0.6, 0.8])
    def test_masses_sum_to_one(self, theta: float) -> None:
        masses = adaptive_outcome_masses(22, theta, 0.6, TAU)
        assert sum(masses) == pytest.approx(1.0, abs=1e-9)
        assert min(masses) >= 0

    def test_masses_sum_to_one_with_min_trials(self) -> None:
        masses = adaptive_outcome_masses(22, 0.8, 0.6, TAU, min_trials=5)
        assert sum(masses) == pytest.approx(1.0, abs=1e-9)

    @pytest.mark.parametrize(
        ("target", "delta", "tau", "expected"),
        [(0.6, 0.1, TAU, 0.80), (0.8, 0.1, 0.9, 0.82), (0.9, 0.05, 0.9, 0.75)],
    )
    def test_known_values_at_max_trials(
        self, target: float, delta: float, tau: float, expected: float
    ) -> None:
        n = max_trials(target, delta, tau)
        assert adaptive_power(n, target + delta, target, tau) == pytest.approx(
            expected, abs=0.01
        )

    def test_at_most_fixed_n_power(self) -> None:
        n = max_trials(0.6, 0.1, TAU)
        k = k_min(n, 0.6, TAU)
        assert k is not None
        fixed = _binom_sf(k, n, 0.7)
        assert adaptive_power(n, 0.7, 0.6, TAU) <= fixed + 1e-12

    def test_min_trials_floor_raises_power(self) -> None:
        n = max_trials(0.6, 0.1, TAU)
        base = adaptive_power(n, 0.7, 0.6, TAU)
        floored = adaptive_power(n, 0.7, 0.6, TAU, min_trials=min_trials(0.6, TAU))
        assert floored >= base

    def test_budget_below_min_trials_never_passes(self) -> None:
        assert adaptive_power(2, 0.99, 0.6, TAU) == 0.0

    def test_power_increases_with_theta(self) -> None:
        powers = [adaptive_power(22, t, 0.6, TAU) for t in (0.4, 0.6, 0.8)]
        assert powers == sorted(powers)


def _resolve(**kw: object):
    args = dict(delta=0.2, min_explicit=None, max_explicit=None)
    args.update(kw)
    return resolve_trial_bounds(0.6, TAU, 3, **args)  # type: ignore[arg-type]


class TestResolveTrialBounds:
    def test_neither_uses_derived(self) -> None:
        plan = _resolve()
        assert plan.derived_min == 3 and plan.derived_max == 24
        assert (plan.min_trials, plan.max_trials) == (3, 24)
        assert plan.power is not None
        assert "N_max=24" in plan.summary()
        assert "adaptive power" in plan.summary()

    def test_default_delta_derives_3_and_69(self) -> None:
        plan = _resolve(delta=0.125)
        assert (plan.min_trials, plan.max_trials) == (3, 69)

    def test_min_only_keeps_derived_max(self) -> None:
        plan = _resolve(min_explicit=8)
        assert (plan.min_trials, plan.max_trials) == (8, 24)
        assert plan.notes == []

    def test_min_below_derived_is_noted_not_error(self) -> None:
        plan = _resolve(min_explicit=1)
        assert plan.min_trials == 1
        assert any("below the derived N_min" in n for n in plan.notes)

    def test_min_above_derived_max_is_error(self) -> None:
        with pytest.raises(ValueError, match="must not exceed"):
            _resolve(min_explicit=25)

    def test_max_only_derives_min(self) -> None:
        plan = _resolve(max_explicit=30)
        assert (plan.min_trials, plan.max_trials) == (3, 30)
        assert plan.warnings == []

    def test_max_below_n_min_is_error(self) -> None:
        with pytest.raises(ValueError, match="PASS is impossible"):
            _resolve(max_explicit=2)

    def test_underpowered_max_warns_with_the_number(self) -> None:
        plan = _resolve(delta=0.1, max_explicit=21)
        assert plan.max_trials == 21
        assert len(plan.warnings) == 1
        assert f"{plan.power:.3f}" in plan.warnings[0]
        assert "underpowered" in plan.warnings[0]

    def test_both_explicit_ok(self) -> None:
        plan = _resolve(min_explicit=5, max_explicit=40)
        assert (plan.min_trials, plan.max_trials) == (5, 40)
        assert plan.warnings == []

    def test_both_explicit_min_above_max_is_error(self) -> None:
        with pytest.raises(ValueError, match="must not exceed"):
            _resolve(min_explicit=12, max_explicit=10)

    def test_both_explicit_max_below_n_min_is_error(self) -> None:
        with pytest.raises(ValueError, match="PASS is impossible"):
            _resolve(min_explicit=0, max_explicit=1)

    def test_both_explicit_underpowered_warns(self) -> None:
        plan = _resolve(delta=0.1, min_explicit=3, max_explicit=20)
        assert plan.warnings

    def test_invalid_delta_is_error(self) -> None:
        with pytest.raises(ValueError, match="delta must be > 0"):
            _resolve(delta=0.0)

    def test_unreachable_derivation_is_error(self) -> None:
        with pytest.raises(ValueError, match=str(MAX_TRIALS_CAP)):
            _resolve(delta=0.001)
