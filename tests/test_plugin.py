"""Unit tests for `binom_eval.plugin` (pytest wiring).

Covers the option registration and its defaults. The fixtures and the
`live_eval` marker themselves are exercised by the per-skill eval suites
that consume them.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from binom_eval import DEFAULT_MAX_TRIALS, DEFAULT_MODEL, DEFAULT_TARGET_RATE, plugin
from binom_eval.stream_json import EvalRun
from binom_eval.grading import BATCH_FLOOR, FAIL_THRESHOLD, PASS_THRESHOLD
from binom_eval.plugin import make_eval_runs_fixture, pytest_addoption


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
        assert "--live-eval-target-rate" in options
        assert "--live-eval-model" in options

    def test_max_trials_defaults_to_constant_and_is_int(self) -> None:
        opt = self._options()["--live-eval-max-trials"]
        assert opt["default"] == DEFAULT_MAX_TRIALS
        assert opt["type"] is int

    def test_target_rate_defaults_to_constant_and_is_float(self) -> None:
        opt = self._options()["--live-eval-target-rate"]
        assert opt["default"] == DEFAULT_TARGET_RATE
        assert opt["type"] is float

    def test_model_defaults_to_constant_and_is_str(self) -> None:
        opt = self._options()["--live-eval-model"]
        assert opt["default"] == DEFAULT_MODEL
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
        model: str | None = None,
    ) -> None:
        self._options = {
            "--live-eval-max-trials": max_trials,
            "--live-eval-target-rate": target,
            "--live-eval-concurrency": concurrency,
            "--live-eval-isolate": isolate,
            "--live-eval-model": model,
        }
        self.pluginmanager = _NullPluginManager()

    def getoption(self, name: str) -> Any:
        return self._options[name]


class TestMakeEvalRunsFixture:
    """The session-scoped fixture the factory builds.

    The wrapped fixture fails fast when the `claude` CLI is absent or
    ANTHROPIC_API_KEY is unset, and otherwise
    runs each eval through `run_eval_adaptive`, returning the runs keyed by
    eval id. `run_eval_adaptive` and `load_evals` are stubbed so no real
    model call happens.
    """

    @staticmethod
    def _fixture_fn() -> Any:
        fixture = make_eval_runs_fixture(
            Path("evals.json"), Path("."), "demo", {}
        )
        return fixture.__wrapped__

    def test_fails_when_claude_cli_is_absent(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
        monkeypatch.setattr(plugin.shutil, "which", lambda _name: None)
        with pytest.raises(pytest.fail.Exception, match="claude CLI not found"):
            self._fixture_fn()(_StubConfig(21, 2.0 / 3.0))

    def test_fails_when_api_key_is_unset(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            plugin.shutil, "which", lambda _name: "/usr/bin/claude"
        )
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        with pytest.raises(pytest.fail.Exception, match="ANTHROPIC_API_KEY"):
            self._fixture_fn()(_StubConfig(21, 2.0 / 3.0))

    def test_builds_runs_keyed_by_eval_id_when_cli_present(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
        monkeypatch.setattr(
            plugin.shutil, "which", lambda _name: "/usr/bin/claude"
        )
        monkeypatch.setattr(
            plugin,
            "load_evals",
            lambda _path, _handlers: [
                {"id": "e1", "assertions": []},
                {"id": "e2", "assertions": []},
            ],
        )
        calls: list[tuple[str, int, float]] = []

        def fake_adaptive(
            item: dict[str, Any],
            repo_root: Path,
            skill_name: str,
            max_trials: int,
            target: float,
            checks: list[Any],
            *,
            gate: Any = None,
            isolate: bool = False,
            model: str | None = None,
        ) -> list[EvalRun]:
            calls.append((item["id"], max_trials, target))
            return [
                EvalRun(
                    eval_id=item["id"],
                    prompt="",
                    skill_invoked=False,
                    assistant_text="",
                )
            ]

        monkeypatch.setattr(plugin, "run_eval_adaptive", fake_adaptive)
        # The pre-flight model probe is covered in test_runner; stub it
        # here so the fixture-wiring test makes no live `claude` call.
        monkeypatch.setattr(plugin, "validate_model", lambda _model: None)

        result = self._fixture_fn()(_StubConfig(21, 2.0 / 3.0))

        assert set(result.keys()) == {"e1", "e2"}
        assert len(result["e1"]) == 1
        assert result["e1"][0].eval_id == "e1"
        assert len(result["e2"]) == 1
        assert result["e2"][0].eval_id == "e2"
        # Evals are driven through a thread pool so recorded order is not
        # guaranteed; sort by id before asserting the shared params.
        sorted_calls = sorted(calls, key=lambda c: c[0])
        assert sorted_calls[0] == ("e1", 21, 2.0 / 3.0)
        assert sorted_calls[1] == ("e2", 21, 2.0 / 3.0)
