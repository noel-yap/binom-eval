"""pytest integration: options, the `live_eval` marker, and the run fixture.

The only module that touches pytest at import time. It registers the
`--live-eval-max-trials` / `--live-eval-target-rate` options and the
`live_eval` marker (both re-exported through `skills/conftest.py` so they
apply once across the whole skills tree) and builds the session-scoped
fixture that runs `claude -p` across adaptive trial batches per eval, graded
by the Beta-binomial verdict in `binom_eval.grading`.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pytest

from binom_eval.grading import (
    FAILURE_SECTION_MAX_CHARS,
    PASS_THRESHOLD,
    _eval_checks,
    load_evals,
    run_eval_adaptive,
)
from binom_eval.runner import resolve_runner
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
# θ ≥ 0.9 -> ~0.2%) while still catching clearly-broken skills; it favours
# not red-flagging working skills over catching mildly-broken (~0.6) ones.
DEFAULT_TARGET_RATE = 3.0 / 5.0

LIVE_EVAL_POSTERIOR_PROPERTY = "live_eval_posterior"


def live_eval_pass_output_enabled(config: pytest.Config) -> bool:
    """True when passing checks should print grading summaries to the
    terminal.
    """
    return config.getoption("--live-eval-verbose") or config.getoption(
        "--live-eval-show-posterior"
    )


def record_live_eval_posterior(node: pytest.Item, summary: str) -> None:
    """Attach a posterior summary to a test node for terminal display."""
    node.user_properties.append((LIVE_EVAL_POSTERIOR_PROPERTY, summary))


class _SessionReporter:
    """Pytest plugin that surfaces the selected backend, CLI version, and
    model in output.

    Registered programmatically in `pytest_configure` so external conftest.py
    files do not need to re-export any additional hooks. The backend label and
    CLI version are captured once at session start from the runner resolved
    from `--live-eval-model`; the model is set by `make_eval_runs_fixture`
    after the runs complete, read from the actual model field in the
    stream-json response.
    """

    def __init__(
        self, config: pytest.Config, *, verbose: bool = False
    ) -> None:
        self._config = config
        self._backend: str = ""
        self._version: str = ""
        self._model: str = ""
        self._verbose = verbose

    def set_backend(self, backend: str, version: str) -> None:
        self._backend = backend
        self._version = version

    def set_model(self, model: str) -> None:
        self._model = model

    def pytest_report_header(self) -> list[str]:
        if self._version:
            return [f"{self._backend} CLI: {self._version}"]
        return []

    def pytest_runtest_logreport(self, report: Any) -> None:
        if (
            not self._verbose
            or report.when != "call"
            or not report.passed
        ):
            return
        lines = [
            value
            for name, value in report.user_properties
            if name == LIVE_EVAL_POSTERIOR_PROPERTY
        ]
        if not lines:
            return
        terminalreporter = self._config.pluginmanager.get_plugin(
            "terminalreporter"
        )
        if terminalreporter is None:
            return
        terminalreporter.ensure_newline()
        for line in lines:
            terminalreporter.write_line(line)

    def pytest_terminal_summary(
        self, terminalreporter: Any, exitstatus: Any, config: Any
    ) -> None:
        parts: list[str] = []
        if self._version:
            parts.append(f"CLI {self._version}")
        if self._model:
            parts.append(f"model {self._model}")
        if parts:
            label = self._backend or "live-eval"
            terminalreporter.write_sep("-", f"{label}: " + "  ".join(parts))


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
            "Target true pass rate a good skill should clear "
            f"(default {DEFAULT_TARGET_RATE:.4f}). The verdict asks how much "
            "posterior mass sits at or above this rate; final grading still "
            "uses the p_good >= 0.5 tiebreak once the trial budget is spent."
        ),
    )
    parser.addoption(
        "--live-eval-pass-threshold",
        action="store",
        type=float,
        default=PASS_THRESHOLD,
        help=(
            "High edge of the verdict band: PASS once p_good exceeds this, "
            "FAIL once p_good drops below its complement (1 minus this). "
            f"Default {PASS_THRESHOLD:.4f} (1 - e^-2); the low edge follows "
            "automatically so the band stays symmetric about 1/2."
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
    parser.addoption(
        "--live-eval-model",
        action="store",
        type=str,
        default=None,
        help=(
            "Required for live evals. Backend and model for every trial, as "
            "`backend:model` (e.g. claude:claude-haiku-4-5-20251001 or "
            "cursor:sonnet-4.5). The `backend:` prefix is mandatory so each "
            "run targets a named harness. Known backends: claude, cursor."
        ),
    )
    parser.addoption(
        "--live-eval-failure-max-chars",
        action="store",
        type=int,
        default=FAILURE_SECTION_MAX_CHARS,
        help=(
            "Per-section character cap when a failing trial's structured "
            "sections are rendered in pytest output. Zero or negative "
            "disables truncation. Default "
            f"{FAILURE_SECTION_MAX_CHARS}."
        ),
    )
    parser.addoption(
        "--live-eval-verbose",
        action="store_true",
        default=False,
        help=(
            "After each passing live-eval check, print full grading detail "
            "(posterior summary plus every trial's assistant reply and tool "
            "uses, using the same layout as failure output but without error "
            "wording). Respects --live-eval-failure-max-chars."
        ),
    )
    parser.addoption(
        "--live-eval-show-posterior",
        action="store_true",
        default=False,
        help=(
            "After each passing live-eval check, print the one-line "
            "posterior summary (P(θ ≥ θ₀ | k, n) and max θ₀). Use "
            "--live-eval-verbose for full per-trial detail."
        ),
    )


def pytest_configure(config: pytest.Config) -> None:
    """Register the `live_eval` marker and the session reporter plugin.

    The reporter's backend label and CLI version come from the runner
    resolved from `--live-eval-model`. Resolution is best-effort here: a bad
    spec leaves the version blank and is reported as a clean failure by the
    `eval_runs` fixture when live evals actually run, rather than crashing
    collection for unrelated (e.g. unit) test runs.
    """
    config.addinivalue_line(
        "markers",
        "live_eval: end-to-end test that invokes a live agent CLI (real model "
        "call). Select with `-m live_eval`; exclude with `-m 'not live_eval'`.",
    )
    reporter = _SessionReporter(
        config,
        verbose=live_eval_pass_output_enabled(config),
    )
    try:
        backend, _model, runner = resolve_runner(
            config.getoption("--live-eval-model")
        )
        reporter.set_backend(backend, runner.version())
    except ValueError:
        pass
    config.pluginmanager.register(reporter, "binom_eval_reporter")


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
        """Run live evals and return parsed runs keyed by eval ID.

        Resolves the backend and model from `--live-eval-model`
        (`backend:model`, bare model = claude), then fails fast with a clear
        message when the spec is malformed, the backend's CLI/credentials are
        missing (`runner.preflight()`), or the model is unusable
        (`runner.validate_model`) -- so a bad setup never silently burns live
        trials.
        """
        spec = pytestconfig.getoption("--live-eval-model")
        try:
            _backend, model, runner = resolve_runner(spec)
        except ValueError as exc:
            pytest.fail(str(exc), pytrace=False)
        preflight_error = runner.preflight()
        if preflight_error is not None:
            pytest.fail(preflight_error, pytrace=False)
        max_trials = pytestconfig.getoption("--live-eval-max-trials")
        target = pytestconfig.getoption("--live-eval-target-rate")
        pass_threshold = pytestconfig.getoption("--live-eval-pass-threshold")
        if not 0.5 < pass_threshold < 1.0:
            pytest.fail(
                "--live-eval-pass-threshold must be strictly between 0.5 "
                f"and 1.0, got {pass_threshold}",
                pytrace=False,
            )
        concurrency = pytestconfig.getoption("--live-eval-concurrency")
        isolate = pytestconfig.getoption("--live-eval-isolate")
        model_error = runner.validate_model(model)
        if model_error is not None:
            pytest.fail(
                f"--live-eval-model {spec!r} is unusable: {model_error}",
                pytrace=False,
            )
        gate = threading.Semaphore(concurrency)
        evals = load_evals(evals_path, assertion_handlers)

        def build(item: dict[str, Any]) -> list[EvalRun]:
            checks = _eval_checks(item, assertion_handlers, skill_name)
            return run_eval_adaptive(
                item,
                repo_root,
                skill_name,
                max_trials,
                target,
                checks,
                pass_threshold=pass_threshold,
                gate=gate,
                isolate=isolate,
                model=model,
                runner=runner,
            )

        # Drive the evals concurrently; the shared `gate` -- not the worker
        # count -- bounds real load, so a few workers per gate slot is plenty
        # to keep it saturated without spawning a thread per eval.
        workers = max(1, min(len(evals), 4 * concurrency))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            runs = list(pool.map(build, evals))
        result = {item["id"]: run for item, run in zip(evals, runs)}

        actual_model = next(
            (run.model for run_list in result.values() for run in run_list if run.model),
            "",
        )
        if actual_model:
            reporter = pytestconfig.pluginmanager.get_plugin(
                "binom_eval_reporter"
            )
            if reporter is not None:
                reporter.set_model(actual_model)

        return result

    return eval_runs


@pytest.fixture(scope="session")
def live_eval_target_rate(pytestconfig: pytest.Config) -> float:
    """The target true pass rate the verdict grades each check against."""
    return pytestconfig.getoption("--live-eval-target-rate")


@pytest.fixture(scope="session")
def live_eval_pass_threshold(pytestconfig: pytest.Config) -> float:
    """High edge of the symmetric verdict band used during adaptive trials."""
    return pytestconfig.getoption("--live-eval-pass-threshold")


@pytest.fixture(scope="session")
def live_eval_failure_max_chars(pytestconfig: pytest.Config) -> int:
    """Per-section character cap for rendered trial-failure sections."""
    return pytestconfig.getoption("--live-eval-failure-max-chars")


@pytest.fixture(scope="session")
def live_eval_verbose(pytestconfig: pytest.Config) -> bool:
    """When true, passing checks print full trial detail to the terminal."""
    return pytestconfig.getoption("--live-eval-verbose")


@pytest.fixture(scope="session")
def live_eval_show_posterior(pytestconfig: pytest.Config) -> bool:
    """When true, passing checks print the one-line posterior summary only."""
    return pytestconfig.getoption("--live-eval-show-posterior")
