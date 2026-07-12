"""Unit tests for `binom_eval.plugin` (pytest wiring).

Covers the option registration and its defaults. The fixtures and the
`live_eval` marker themselves are exercised by the per-skill eval suites
that consume them.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from binom_eval import (
    DEFAULT_MAX_TRIALS,
    DEFAULT_MIN_TRIALS,
    DEFAULT_TARGET_RATE,
    plugin,
)
from binom_eval.stream_json import EvalRun
from binom_eval.grading import (
    BATCH_FLOOR,
    FAIL_THRESHOLD,
    FAILURE_SECTION_MAX_CHARS,
    PASS_THRESHOLD,
)
from binom_eval.plugin import (
    LIVE_EVAL_POSTERIOR_PROPERTY,
    make_eval_runs_fixture,
    pytest_addoption,
    record_live_eval_posterior,
)


class _StubParser:
    """Captures `addoption` calls so we can assert names and defaults."""

    def __init__(self) -> None:
        self.options: dict[str, dict[str, Any]] = {}

    def addoption(self, name: str, **kwargs: Any) -> None:
        self.options[name] = kwargs


class TestPytestAddOption:
    def _options(self) -> dict[str, dict[str, Any]]:
        parser = _StubParser()
        pytest_addoption(parser)
        return parser.options

    def test_registers_both_live_eval_options(self) -> None:
        options = self._options()
        assert "--live-eval-max-trials" in options
        assert "--live-eval-min-trials" in options
        assert "--live-eval-target-rate" in options
        assert "--live-eval-pass-threshold" in options
        assert "--live-eval-model" in options
        assert "--live-eval-failure-max-chars" in options
        assert "--live-eval-verbose" in options
        assert "--live-eval-show-posterior" in options

    def test_max_trials_defaults_to_constant_and_is_int(self) -> None:
        opt = self._options()["--live-eval-max-trials"]
        assert opt["default"] == DEFAULT_MAX_TRIALS
        assert opt["type"] is int

    def test_min_trials_defaults_to_constant_and_is_int(self) -> None:
        opt = self._options()["--live-eval-min-trials"]
        assert opt["default"] == DEFAULT_MIN_TRIALS
        assert opt["type"] is int

    def test_target_rate_defaults_to_constant_and_is_float(self) -> None:
        opt = self._options()["--live-eval-target-rate"]
        assert opt["default"] == DEFAULT_TARGET_RATE
        assert opt["type"] is float

    def test_pass_threshold_defaults_to_constant_and_is_float(self) -> None:
        opt = self._options()["--live-eval-pass-threshold"]
        assert opt["default"] == PASS_THRESHOLD
        assert opt["type"] is float

    def test_failure_max_chars_defaults_to_constant_and_is_int(self) -> None:
        opt = self._options()["--live-eval-failure-max-chars"]
        assert opt["default"] == FAILURE_SECTION_MAX_CHARS
        assert opt["type"] is int

    def test_verbose_defaults_to_false(self) -> None:
        opt = self._options()["--live-eval-verbose"]
        assert opt["default"] is False
        assert opt["action"] == "store_true"

    def test_show_posterior_defaults_to_false(self) -> None:
        opt = self._options()["--live-eval-show-posterior"]
        assert opt["default"] is False
        assert opt["action"] == "store_true"

    def test_model_has_no_default_and_is_str(self) -> None:
        # No default: the `backend:` prefix is mandatory, so a live run must
        # name its harness explicitly rather than fall back to one.
        opt = self._options()["--live-eval-model"]
        assert opt["default"] is None
        assert opt["type"] is str


class TestDefaults:
    def test_target_rate_sits_inside_the_unit_interval(self) -> None:
        assert 0.0 < DEFAULT_TARGET_RATE < 1.0

    def test_max_trials_is_a_whole_number_of_floored_batches(self) -> None:
        # 21 = 3 * 7, so the worst case is a clean run of BATCH_FLOOR-sized
        # rounds with no ragged remainder.
        assert DEFAULT_MAX_TRIALS % BATCH_FLOOR == 0

    def test_band_is_symmetric_about_one_half(self) -> None:
        assert abs((PASS_THRESHOLD + FAIL_THRESHOLD) - 1.0) < 1e-12


class _NullPluginManager:
    """Stub for pluginmanager: always returns None from get_plugin."""

    def get_plugin(self, _name: str) -> None:
        return None


class _StubConfig:
    """Stands in for `pytest.Config`, answering `getoption` from a dict."""

    def __init__(
        self,
        max_trials: int,
        target: float,
        concurrency: int = 4,
        isolate: bool = False,
        model: str | None = "claude:haiku",
        pass_threshold: float = PASS_THRESHOLD,
        min_trials: int = DEFAULT_MIN_TRIALS,
    ) -> None:
        self._options = {
            "--live-eval-max-trials": max_trials,
            "--live-eval-min-trials": min_trials,
            "--live-eval-target-rate": target,
            "--live-eval-pass-threshold": pass_threshold,
            "--live-eval-concurrency": concurrency,
            "--live-eval-isolate": isolate,
            "--live-eval-model": model,
        }
        self.pluginmanager = _NullPluginManager()

    def getoption(self, name: str) -> Any:
        return self._options[name]


class _FakeRunner:
    """A stand-in backend: canned `preflight`/`validate_model` verdicts.

    The fixture only calls `preflight()` and `validate_model()` on the runner
    before handing it to the (stubbed) driver, so this needs nothing more.
    """

    def __init__(
        self,
        preflight: str | None = None,
        model_error: str | None = None,
    ) -> None:
        self._preflight = preflight
        self._model_error = model_error

    def preflight(self) -> str | None:
        return self._preflight

    def validate_model(self, _model: str, _timeout: int = 30) -> str | None:
        return self._model_error


class TestMakeEvalRunsFixture:
    """The session-scoped fixture the factory builds.

    The wrapped fixture fails fast on a malformed `--live-eval-model` spec,
    when the backend's `preflight()` reports a missing CLI/credential, or when
    `validate_model` rejects the model; otherwise it runs each eval through
    `run_eval_adaptive`, returning the runs keyed by eval id. `resolve_runner`,
    `run_eval_adaptive`, and `load_evals` are stubbed so no real model call
    happens.
    """

    @staticmethod
    def _fixture_fn(**kwargs: Any) -> Any:
        fixture = make_eval_runs_fixture(
            Path("evals.json"), Path("."), "demo", {}, **kwargs
        )
        return fixture.__wrapped__

    def test_fails_on_unknown_backend(self) -> None:
        with pytest.raises(pytest.fail.Exception, match="unknown eval backend"):
            self._fixture_fn()(_StubConfig(21, 2.0 / 3.0, model="gpt:4o"))

    def test_fails_when_model_missing(self) -> None:
        with pytest.raises(
            pytest.fail.Exception, match="must be 'backend:model'"
        ):
            self._fixture_fn()(_StubConfig(21, 2.0 / 3.0, model=None))

    def test_fails_when_prefix_omitted(self) -> None:
        with pytest.raises(
            pytest.fail.Exception, match="must be 'backend:model'"
        ):
            self._fixture_fn()(_StubConfig(21, 2.0 / 3.0, model="haiku"))

    def test_fails_when_preflight_reports_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        runner = _FakeRunner(preflight="claude CLI not found on PATH")
        monkeypatch.setattr(
            plugin, "resolve_runner", lambda _spec: ("claude", "m", runner)
        )
        with pytest.raises(pytest.fail.Exception, match="claude CLI not found"):
            self._fixture_fn()(_StubConfig(21, 2.0 / 3.0))

    def test_fails_when_model_is_unusable(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        runner = _FakeRunner(model_error="model not found: bad")
        monkeypatch.setattr(
            plugin, "resolve_runner", lambda _spec: ("claude", "bad", runner)
        )
        with pytest.raises(pytest.fail.Exception, match="is unusable"):
            self._fixture_fn()(_StubConfig(21, 2.0 / 3.0, model="claude:bad"))

    def test_fails_when_pass_threshold_not_above_half(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            plugin,
            "resolve_runner",
            lambda _spec: ("claude", "m", _FakeRunner()),
        )
        with pytest.raises(
            pytest.fail.Exception, match="strictly between 0.5"
        ):
            self._fixture_fn()(
                _StubConfig(21, 2.0 / 3.0, pass_threshold=0.5)
            )

    def test_fails_when_pass_threshold_reaches_one(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            plugin,
            "resolve_runner",
            lambda _spec: ("claude", "m", _FakeRunner()),
        )
        with pytest.raises(
            pytest.fail.Exception, match="strictly between 0.5"
        ):
            self._fixture_fn()(
                _StubConfig(21, 2.0 / 3.0, pass_threshold=1.0)
            )

    def test_fails_when_min_trials_exceeds_max_trials(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            plugin,
            "resolve_runner",
            lambda _spec: ("claude", "m", _FakeRunner()),
        )
        with pytest.raises(
            pytest.fail.Exception, match="must not exceed"
        ):
            self._fixture_fn()(
                _StubConfig(10, 2.0 / 3.0, min_trials=11)
            )

    def test_builds_runs_keyed_by_eval_id_when_cli_present(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            plugin, "resolve_runner", lambda _spec: ("claude", "m", _FakeRunner())
        )
        monkeypatch.setattr(
            plugin,
            "load_evals",
            lambda _path, _handlers: [
                {"id": "e1", "assertions": []},
                {"id": "e2", "assertions": []},
            ],
        )
        calls: list[tuple[str, int, float, float]] = []

        def fake_adaptive(
            item: dict[str, Any],
            repo_root: Path,
            skill_name: str,
            max_trials: int,
            target: float,
            checks: list[Any],
            *,
            pass_threshold: float = PASS_THRESHOLD,
            min_trials: int = 0,
            gate: Any = None,
            isolate: bool = False,
            model: str,
            runner: Any = None,
        ) -> list[EvalRun]:
            calls.append((item["id"], max_trials, target, pass_threshold))
            return [
                EvalRun(
                    eval_id=item["id"],
                    prompt="",
                    skill_invoked=False,
                    assistant_text="",
                )
            ]

        result = self._fixture_fn(run_adaptive=fake_adaptive)(
            _StubConfig(21, 2.0 / 3.0)
        )

        assert set(result.keys()) == {"e1", "e2"}
        assert len(result["e1"]) == 1
        assert result["e1"][0].eval_id == "e1"
        assert len(result["e2"]) == 1
        assert result["e2"][0].eval_id == "e2"
        # Evals are driven through a thread pool so recorded order is not
        # guaranteed; sort by id before asserting the shared params.
        sorted_calls = sorted(calls, key=lambda c: c[0])
        assert sorted_calls[0] == ("e1", 21, 2.0 / 3.0, PASS_THRESHOLD)
        assert sorted_calls[1] == ("e2", 21, 2.0 / 3.0, PASS_THRESHOLD)

    def test_forwards_custom_pass_threshold_to_adaptive(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            plugin, "resolve_runner", lambda _spec: ("claude", "m", _FakeRunner())
        )
        monkeypatch.setattr(
            plugin,
            "load_evals",
            lambda _path, _handlers: [{"id": "e1", "assertions": []}],
        )
        custom = 0.95
        calls: list[float] = []

        def fake_adaptive(
            item: dict[str, Any],
            repo_root: Path,
            skill_name: str,
            max_trials: int,
            target: float,
            checks: list[Any],
            *,
            pass_threshold: float = PASS_THRESHOLD,
            min_trials: int = 0,
            gate: Any = None,
            isolate: bool = False,
            model: str,
            runner: Any = None,
        ) -> list[EvalRun]:
            calls.append(pass_threshold)
            return [
                EvalRun(
                    eval_id=item["id"],
                    prompt="",
                    skill_invoked=False,
                    assistant_text="",
                )
            ]

        self._fixture_fn(run_adaptive=fake_adaptive)(
            _StubConfig(21, 2.0 / 3.0, pass_threshold=custom)
        )

        assert calls == [custom]

    def test_forwards_custom_min_trials_to_adaptive(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            plugin,
            "resolve_runner",
            lambda _spec: ("claude", "m", _FakeRunner()),
        )
        monkeypatch.setattr(
            plugin,
            "load_evals",
            lambda _path, _handlers: [{"id": "e1", "assertions": []}],
        )
        custom = 7
        calls: list[int] = []

        def fake_adaptive(
            item: dict[str, Any],
            repo_root: Path,
            skill_name: str,
            max_trials: int,
            target: float,
            checks: list[Any],
            *,
            pass_threshold: float = PASS_THRESHOLD,
            min_trials: int = 0,
            gate: Any = None,
            isolate: bool = False,
            model: str,
            runner: Any = None,
        ) -> list[EvalRun]:
            calls.append(min_trials)
            return [
                EvalRun(
                    eval_id=item["id"],
                    prompt="",
                    skill_invoked=False,
                    assistant_text="",
                )
            ]

        self._fixture_fn(run_adaptive=fake_adaptive)(
            _StubConfig(21, 2.0 / 3.0, min_trials=custom)
        )

        assert calls == [custom]


class _StubTerminalReporter:
    """Collects the lines the posterior reporter writes."""

    def __init__(self) -> None:
        self.lines: list[str] = []

    def ensure_newline(self) -> None:
        pass

    def write_line(self, line: str, **kwargs: Any) -> None:
        self.lines.append(line)


class _StubPluginManager:
    def __init__(self, terminalreporter: Any) -> None:
        self._terminalreporter = terminalreporter

    def get_plugin(self, name: str) -> Any:
        return self._terminalreporter


class _StubReporterConfig:
    """Config stub exposing just the pluginmanager the reporter uses."""

    def __init__(self, terminalreporter: Any) -> None:
        self.pluginmanager = _StubPluginManager(terminalreporter)


class _StubReport:
    def __init__(
        self,
        user_properties: list[tuple[str, str]],
        *,
        when: str = "call",
        passed: bool = True,
    ) -> None:
        self.user_properties = user_properties
        self.when = when
        self.passed = passed


class TestPosteriorReporting:
    def test_record_live_eval_posterior_attaches_user_property(self) -> None:
        class _Node:
            def __init__(self) -> None:
                self.user_properties: list[tuple[str, str]] = []

        summary = "demo::check: P(θ ≥ 0.600 | k=3, n=3) = 0.800"
        node = _Node()
        record_live_eval_posterior(node, summary)
        assert node.user_properties == [
            (LIVE_EVAL_POSTERIOR_PROPERTY, summary)
        ]

    def test_runtest_logreport_prints_posterior_on_pass(self) -> None:
        terminal = _StubTerminalReporter()
        reporter = plugin._SessionReporter(
            _StubReporterConfig(terminal), verbose=True
        )
        summary = "demo: P(θ ≥ 0.600 | k=3, n=3) = 0.800"
        report = _StubReport([(LIVE_EVAL_POSTERIOR_PROPERTY, summary)])

        reporter.pytest_runtest_logreport(report)

        assert terminal.lines == [summary]

    def test_runtest_logreport_skips_when_disabled(self) -> None:
        terminal = _StubTerminalReporter()
        reporter = plugin._SessionReporter(
            _StubReporterConfig(terminal), verbose=False
        )
        report = _StubReport([(LIVE_EVAL_POSTERIOR_PROPERTY, "hidden")])

        reporter.pytest_runtest_logreport(report)

        assert terminal.lines == []

    def test_runtest_logreport_skips_setup_phase(self) -> None:
        terminal = _StubTerminalReporter()
        reporter = plugin._SessionReporter(
            _StubReporterConfig(terminal), verbose=True
        )
        report = _StubReport(
            [(LIVE_EVAL_POSTERIOR_PROPERTY, "early")], when="setup"
        )

        reporter.pytest_runtest_logreport(report)

        assert terminal.lines == []

    def test_runtest_logreport_skips_failed_test(self) -> None:
        terminal = _StubTerminalReporter()
        reporter = plugin._SessionReporter(
            _StubReporterConfig(terminal), verbose=True
        )
        report = _StubReport(
            [(LIVE_EVAL_POSTERIOR_PROPERTY, "nope")], passed=False
        )

        reporter.pytest_runtest_logreport(report)

        assert terminal.lines == []

    def test_runtest_logreport_ignores_unrelated_properties(
        self,
    ) -> None:
        terminal = _StubTerminalReporter()
        reporter = plugin._SessionReporter(
            _StubReporterConfig(terminal), verbose=True
        )
        report = _StubReport([("some_other_property", "x")])

        reporter.pytest_runtest_logreport(report)

        assert terminal.lines == []

    def test_runtest_logreport_tolerates_missing_terminal(self) -> None:
        reporter = plugin._SessionReporter(
            _StubReporterConfig(None), verbose=True
        )
        report = _StubReport(
            [(LIVE_EVAL_POSTERIOR_PROPERTY, "orphan")]
        )

        reporter.pytest_runtest_logreport(report)  # must not raise


class TestLiveEvalPassOutputEnabled:
    class _StubConfig:
        def __init__(self, **flags: bool) -> None:
            self._flags = flags

        def getoption(self, name: str) -> bool:
            return self._flags.get(name, False)

    def test_disabled_by_default(self) -> None:
        config = self._StubConfig()
        assert plugin.live_eval_pass_output_enabled(config) is False

    def test_verbose_enables_output(self) -> None:
        config = self._StubConfig(**{"--live-eval-verbose": True})
        assert plugin.live_eval_pass_output_enabled(config) is True

    def test_show_posterior_enables_output(self) -> None:
        config = self._StubConfig(**{"--live-eval-show-posterior": True})
        assert plugin.live_eval_pass_output_enabled(config) is True
