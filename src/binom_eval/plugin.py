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
import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
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

# How many `claude -p` runs may be in flight at once across the whole session.
# Evals are driven in parallel and each fans its trials out too, so this single
# ceiling -- enforced by one shared semaphore threaded through every run --
# bounds total live calls regardless of suite size, keeping local load and API
# rate pressure in check. Sitting just above `BATCH_FLOOR` (3), it leaves room
# for a second eval to keep the gate warm through another's re-grading gap
# rather than letting one eval's opening batch monopolize it. Raise it to
# finish faster when the API and machine can take it; drop it to 1 to run
# fully serially.
DEFAULT_CONCURRENCY = 5

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
    parser.addoption(
        "--live-eval-concurrency",
        action="store",
        type=int,
        default=DEFAULT_CONCURRENCY,
        help=(
            "Maximum `claude -p` runs in flight at once across the whole "
            "session, shared by parallel evals and their trial batches alike. "
            f"Default {DEFAULT_CONCURRENCY}; set to 1 to run fully serially."
        ),
    )
    parser.addoption(
        "--live-eval-isolate",
        action="store_true",
        default=False,
        help=(
            "Run each `claude -p` trial in a throwaway copy of the skill's "
            "repo root instead of the shared tree. Needed for skills that "
            "write to the working tree so concurrent runs cannot clobber each "
            "other; off by default since it copies the tree per run."
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
    value is a list of `EvalRun` per eval (one per trial run). The evals are
    driven in parallel and each runs its trials in adaptive concurrent batches
    that stop as soon as the Beta-binomial verdict locks, decided from
    `assertion_handlers` (plus the skill-trigger check). A single shared
    semaphore (`--live-eval-concurrency`) caps total live calls across all of
    this; `--live-eval-isolate` runs each trial in a throwaway copy of
    `repo_root` for skills that write to the tree. Every run is a fresh live
    call; results are never cached.
    """

    @pytest.fixture(scope="session")
    def eval_runs(pytestconfig: pytest.Config) -> dict[str, list[EvalRun]]:
        if shutil.which("claude") is None:
            pytest.skip("claude CLI not found on PATH")
        max_trials = pytestconfig.getoption("--live-eval-max-trials")
        target = pytestconfig.getoption("--live-eval-target-rate")
        concurrency = pytestconfig.getoption("--live-eval-concurrency")
        isolate = pytestconfig.getoption("--live-eval-isolate")
        gate = threading.Semaphore(concurrency)
        evals = load_evals(evals_path, assertion_handlers)

        def build(item: dict[str, Any]) -> list[EvalRun]:
            checks = _eval_checks(item, assertion_handlers)
            return run_eval_adaptive(
                item,
                repo_root,
                skill_name,
                max_trials,
                target,
                checks,
                gate=gate,
                isolate=isolate,
            )

        # Drive the evals concurrently; the shared `gate` -- not the worker
        # count -- bounds real load, so a few workers per gate slot is plenty
        # to keep it saturated without spawning a thread per eval.
        workers = max(1, min(len(evals), 4 * concurrency))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            runs = list(pool.map(build, evals))
        return {item["id"]: run for item, run in zip(evals, runs)}

    return eval_runs


@pytest.fixture(scope="session")
def live_eval_target_rate(pytestconfig: pytest.Config) -> float:
    """The target true pass rate the verdict grades each check against."""
    return pytestconfig.getoption("--live-eval-target-rate")