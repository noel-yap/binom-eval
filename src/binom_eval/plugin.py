"""pytest integration: options, the `live_eval` marker, and the run fixture.

The only module that touches pytest at import time. It registers the
`--live-eval-max-trials` / `--live-eval-target-rate` options and the
`live_eval` marker (both re-exported through `skills/conftest.py` so they
apply once across the whole skills tree) and builds the session-scoped
fixture that runs `claude -p` across adaptive trial batches per eval, graded
by the Beta-binomial verdict in `binom_eval.grading`.
"""

from __future__ import annotations

import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from binom_eval.grading import (
    PASS_THRESHOLD,
    _eval_checks,
    load_evals,
    run_eval_adaptive,
)
from binom_eval.stream_json import EvalRun

# Budget ceiling: the most trials any single eval will ever run. A verdict
# usually locks well before this; it only bites for skills sitting right at
# the target rate, which are genuinely undecidable. 21 = 3 * 7 divides evenly
# by BATCH_FLOOR, so the worst case is a clean seven rounds of three.
DEFAULT_MAX_TRIALS = 21

# The true pass rate a good skill should clear. The verdict asks how much
# posterior mass sits at or above this. 3/5 ("passes at least three of every
# five attempts") keeps false fails on genuinely-good skills very rare (true
# rate >= 0.9 -> ~0.2%) while still catching clearly-broken skills; it favours
# not red-flagging working skills over catching mildly-broken (~0.6) ones.
DEFAULT_TARGET_RATE = 3.0 / 5.0


def pytest_addoption(parser: pytest.Parser) -> None:
    """Register the `--live-eval-max-trials` and `--live-eval-target-rate`
    options.

    Re-exported through `skills/conftest.py` so the budget ceiling and the
    target pass rate can be tuned from the pytest command line.
    """
    parser.addoption(
        "--live-eval-max-trials",
        action="store",
        type=int,
        default=DEFAULT_MAX_TRIALS,
        help=(
            "Budget ceiling: the most times any eval is run before the "
            "verdict is forced. Trials usually stop sooner once the "
            f"posterior locks. Default {DEFAULT_MAX_TRIALS}."
        ),
    )
    parser.addoption(
        "--live-eval-target-rate",
        action="store",
        type=float,
        default=DEFAULT_TARGET_RATE,
        help=(
            "Target true pass rate a good skill should clear. The verdict "
            "PASSes once the posterior puts > %.3f of its mass at or above "
            "this rate, FAILs once < %.3f. Default %.4f."
            % (PASS_THRESHOLD, 1.0 - PASS_THRESHOLD, DEFAULT_TARGET_RATE)
        ),
    )


def pytest_configure(config: pytest.Config) -> None:
    """Register the `live_eval` marker for selecting/skipping live tests."""
    config.addinivalue_line(
        "markers",
        "live_eval: end-to-end test that invokes `claude -p` (real model "
        "call). Select with `-m live_eval`; exclude with `-m 'not live_eval'`.",
    )


def make_eval_runs_fixture(
    evals_path: Path,
    repo_root: Path,
    skill_name: str,
    assertion_handlers: dict[str, Callable[[EvalRun], None]],
) -> Callable[..., dict[str, list[EvalRun]]]:
    """Build a session-scoped pytest fixture that runs claude -p up to
    `--live-eval-max-trials` times per eval in `evals_path` and returns the
    parsed runs keyed by eval id.

    Per-skill conftest.py binds the returned fixture to the name
    `eval_runs` so per-skill `test_evals.py` can request it directly. The
    value is a list of `EvalRun` per eval (one per trial run): trials run
    in adaptive concurrent batches that stop as soon as the Beta-binomial
    verdict locks, decided from `assertion_handlers` (plus the skill-trigger
    check). Every run is a fresh live call; results are never cached.
    """

    @pytest.fixture(scope="session")
    def eval_runs(pytestconfig: pytest.Config) -> dict[str, list[EvalRun]]:
        pytest.skip("claude CLI not found on PATH") if shutil.which(
            "claude"
        ) is None else None
        max_trials = pytestconfig.getoption("--live-eval-max-trials")
        target = pytestconfig.getoption("--live-eval-target-rate")

        def build(item: dict[str, Any]) -> list[EvalRun]:
            checks = _eval_checks(item, assertion_handlers)
            return run_eval_adaptive(
                item, repo_root, skill_name, max_trials, target, checks
            )

        return {
            item["id"]: build(item)
            for item in load_evals(evals_path, assertion_handlers)
        }

    return eval_runs


@pytest.fixture(scope="session")
def live_eval_target_rate(pytestconfig: pytest.Config) -> float:
    """The target true pass rate the verdict grades each check against."""
    return pytestconfig.getoption("--live-eval-target-rate")