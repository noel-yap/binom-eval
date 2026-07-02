"""Unit tests for `binom_eval.grading` (Bayesian verdict + rollups).

Covers the Beta-binomial core (`_betainc`, `posterior_pass_prob`,
`_verdict`, `eval_passed`), the adaptive trial driver (`next_batch_size`,
`run_eval_adaptive` and the checks feeding them), and the rollups per-skill
suites grade with (`trial_outcomes`, `trial_outcomes_passed`,
`trial_outcomes_failure_message`, `failing_assertions`, `trigger_pass_counts`). `binom_eval` is
skill-independent, so this logic is tested once here rather than duplicated
per skill.

Expected numbers below pin the mechanics against a fixed representative
`TARGET` of 2/3 (deliberately independent of the shipped default, so a
default retune cannot break these unit tests), with the band (e^-2, 1 - e^-2),
the prior Beta(1, 1), and `BATCH_FLOOR` of 3.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import binom_eval
from binom_eval import (
    BATCH_FLOOR,
    FAILURE_SECTION_MAX_CHARS,
    FAIL_THRESHOLD,
    PASS_THRESHOLD,
    AssertionFailure,
    EvalRun,
    TrialFailure,
    _check_failures,
    _eval_checks,
    _trigger_check,
    assert_handler_coverage,
    eval_passed,
    expand_eval_item,
    expand_evals,
    failing_assertions,
    load_evals,
    next_batch_size,
    posterior_pass_prob,
    trial_outcomes,
    trial_outcomes_failure_message,
    trial_outcomes_passed,
    trigger_pass_counts,
)
from binom_eval.grading import Verdict, _betainc, _resolve_shortfall, _verdict

TARGET = 2.0 / 3.0


def _runs(*passed: bool) -> list[EvalRun]:
    """Trials whose `skill_invoked` flag stands in for pass/fail."""
    return [
        EvalRun(eval_id="t", prompt="", skill_invoked=p, assistant_text="")
        for p in passed
    ]


def _skill_check(run: EvalRun) -> None:
    if not run.skill_invoked:
        raise AssertionFailure("miss")


class TestBetainc:
    """The regularized incomplete beta -- the Beta CDF underneath everything."""

    # Beta(1, 1) is uniform, so its CDF at x is x.
    @pytest.mark.parametrize("x", [0.0, 0.3, 0.5, 0.864, 1.0])
    def test_uniform_cdf_is_identity(self, x: float) -> None:
        assert _betainc(x, 1.0, 1.0) == pytest.approx(x, abs=1e-9)

    def test_clamps_outside_unit_interval(self) -> None:
        assert _betainc(-0.1, 2.0, 3.0) == 0.0
        assert _betainc(1.5, 2.0, 3.0) == 1.0

    def test_matches_closed_form_for_small_beta(self) -> None:
        # For Beta(1, b), P(theta <= x) = 1 - (1 - x)**b.
        assert _betainc(0.5, 1.0, 3.0) == pytest.approx(1 - 0.5**3, abs=1e-9)


class TestPosteriorPassProb:
    """p_good = P(theta >= target) under the Beta(1,1) posterior."""

    def test_no_trials_is_prior_mass_above_target(self) -> None:
        # Beta(1, 1): P(theta >= t) = 1 - t.
        assert posterior_pass_prob(0, 0, TARGET) == pytest.approx(
            1 - TARGET, abs=1e-9
        )

    def test_rises_with_more_passes(self) -> None:
        assert (
            posterior_pass_prob(1, 1, TARGET)
            < posterior_pass_prob(3, 3, TARGET)
            < posterior_pass_prob(6, 6, TARGET)
        )

    def test_falls_with_more_failures(self) -> None:
        assert (
            posterior_pass_prob(0, 1, TARGET)
            > posterior_pass_prob(0, 3, TARGET)
            > posterior_pass_prob(0, 6, TARGET)
        )


class TestVerdict:
    """The band that drives early stopping."""

    def test_undetermined_before_any_trials(self) -> None:
        assert _verdict(0, 0, TARGET) == Verdict.UNDETERMINED

    def test_pass_once_above_high_edge(self) -> None:
        # 6 clean passes clears PASS_THRESHOLD at target 2/3.
        assert _verdict(6, 6, TARGET) == Verdict.PASS
        assert posterior_pass_prob(6, 6, TARGET) > PASS_THRESHOLD

    def test_fail_once_below_low_edge(self) -> None:
        # 2 clean failures drops below FAIL_THRESHOLD at target 2/3.
        assert _verdict(0, 2, TARGET) == Verdict.FAIL
        assert posterior_pass_prob(0, 2, TARGET) < FAIL_THRESHOLD


class TestEvalPassed:
    """The final grade: posterior majority above the bar."""

    def test_passes_when_majority_mass_above_bar(self) -> None:
        assert eval_passed(3, 3, TARGET) is True  # p_good ~0.80

    def test_fails_when_majority_mass_below_bar(self) -> None:
        assert eval_passed(2, 3, TARGET) is False
        assert eval_passed(4, 6, TARGET) is False


class TestTriggerCheck:
    def test_passes_when_skill_invoked(self) -> None:
        _trigger_check(_runs(True)[0])  # should not raise

    def test_raises_when_skill_not_invoked(self) -> None:
        with pytest.raises(AssertionError, match="skill was not invoked"):
            _trigger_check(_runs(False)[0])


class TestTrialOutcomes:
    def test_records_pass_and_fail_per_trial(self) -> None:
        def check(run: EvalRun) -> None:
            if not run.skill_invoked:
                raise AssertionFailure("miss")

        outcomes = trial_outcomes(_runs(True, False), check)
        assert outcomes[0] == (0, None)
        assert outcomes[1][0] == 1
        assert outcomes[1][1] == TrialFailure("miss")

    def test_captures_assertion_failure_sections(self) -> None:
        def check(run: EvalRun) -> None:
            raise AssertionFailure(
                "structured miss",
                sections=(("Input", "before"), ("Output", run.assistant_text)),
            )

        outcomes = trial_outcomes(
            [EvalRun(eval_id="t", prompt="", skill_invoked=True, assistant_text="after")],
            check,
        )
        assert outcomes[0][1] == TrialFailure(
            "structured miss",
            sections=(("Input", "before"), ("Output", "after")),
        )


class TestTrialOutcomesGrading:
    def test_passes_when_posterior_clears_bar(self) -> None:
        # 3 of 3 passes -> p_good ~0.80 >= 0.5.
        outcomes = [(0, None), (1, None), (2, None)]
        assert trial_outcomes_passed(outcomes, TARGET), (
            trial_outcomes_failure_message(outcomes, TARGET, "x")
        )

    def test_fails_when_posterior_below_bar(self) -> None:
        outcomes = [
            (0, None),
            (1, TrialFailure("bad")),
            (2, TrialFailure("bad")),
        ]
        with pytest.raises(AssertionError, match=r"1/3 trials passed"):
            assert trial_outcomes_passed(outcomes, TARGET), (
                trial_outcomes_failure_message(outcomes, TARGET, "x")
            )

    def test_failure_message_reports_p_good(self) -> None:
        outcomes = [(0, TrialFailure("bad")), (1, TrialFailure("bad"))]
        with pytest.raises(AssertionError, match=r"P\(rate >= 0\.667\)"):
            assert trial_outcomes_passed(outcomes, TARGET), (
                trial_outcomes_failure_message(outcomes, TARGET, "x")
            )

    def test_failure_message_renders_structured_sections(self) -> None:
        outcomes = [
            (
                0,
                TrialFailure(
                    "isCacheStale not in output",
                    sections=(
                        ("Input", "export function isCacheStale() {}"),
                        ("Output", "export function hasCacheExpired() {}"),
                    ),
                ),
            )
        ]
        message = trial_outcomes_failure_message(outcomes, TARGET, "demo::check")
        assert "trial 0: isCacheStale not in output" in message
        assert "Input:" in message
        assert "export function isCacheStale() {}" in message
        assert "Output:" in message
        assert "export function hasCacheExpired() {}" in message

    @staticmethod
    def _long_body_outcomes(body: str) -> list[tuple[int, TrialFailure]]:
        return [(0, TrialFailure("miss", sections=(("Output", body),)))]

    def test_failure_message_truncates_long_sections_by_default(self) -> None:
        body = "x" * (FAILURE_SECTION_MAX_CHARS + 500)
        message = trial_outcomes_failure_message(
            self._long_body_outcomes(body), TARGET, "demo::check"
        )
        assert "... (500 chars truncated)" in message
        assert body not in message

    def test_failure_message_honors_max_chars_override(self) -> None:
        body = "x" * (FAILURE_SECTION_MAX_CHARS + 500)
        message = trial_outcomes_failure_message(
            self._long_body_outcomes(body),
            TARGET,
            "demo::check",
            max_chars=len(body),
        )
        assert body in message
        assert "chars truncated" not in message

    def test_failure_message_zero_max_chars_disables_truncation(self) -> None:
        body = "x" * (FAILURE_SECTION_MAX_CHARS + 500)
        message = trial_outcomes_failure_message(
            self._long_body_outcomes(body), TARGET, "demo::check", max_chars=0
        )
        assert body in message
        assert "chars truncated" not in message


class TestFailingAssertions:
    """Per-eval rollup of which assertions failed the posterior bar."""

    @staticmethod
    def _skill(run: EvalRun) -> None:
        if not run.skill_invoked:
            raise AssertionFailure("skill")

    @staticmethod
    def _text(run: EvalRun) -> None:
        if not run.assistant_text:
            raise AssertionFailure("text")

    def _handlers(self) -> dict:
        return {"a": self._skill, "b": self._text}

    def test_empty_when_all_clear_bar(self) -> None:
        runs = _runs(True, True, True)  # 3/3 -> p_good ~0.80
        assertions = [{"id": "a"}]
        assert (
            failing_assertions(runs, assertions, self._handlers(), TARGET)
            == []
        )

    def test_reports_id_counts_and_p_good_below_bar(self) -> None:
        runs = _runs(True, False, False)  # 1/3 invoked the skill
        assertions = [{"id": "a"}]
        result = failing_assertions(runs, assertions, self._handlers(), TARGET)
        assert len(result) == 1
        aid, passes, trials, p_good = result[0]
        assert (aid, passes, trials) == ("a", 1, 3)
        assert p_good < 0.5

    def test_raises_on_assertion_without_a_handler(self) -> None:
        # Coverage is validated at load time; reaching grading with a
        # handlerless assertion is a bug, so it hard-fails rather than skips.
        runs = _runs(False, False)
        assertions = [{"id": "missing"}, {"id": "a"}]
        with pytest.raises(KeyError, match="missing"):
            failing_assertions(runs, assertions, self._handlers(), TARGET)

    def test_collects_every_failing_assertion(self) -> None:
        # All runs miss skill (a) and have empty text (b): both fail the bar.
        runs = _runs(False, False)
        assertions = [{"id": "a"}, {"id": "b"}]
        result = failing_assertions(runs, assertions, self._handlers(), TARGET)
        assert [aid for aid, *_ in result] == ["a", "b"]


class TestAssertHandlerCoverage:
    """Load-time guard that every assertion has a registered handler."""

    @staticmethod
    def _handlers() -> dict:
        return {"a": lambda _r: None, "b": lambda _r: None}

    def test_passes_when_every_assertion_is_handled(self) -> None:
        evals = [{"id": "e1", "assertions": [{"id": "a"}, {"id": "b"}]}]
        assert_handler_coverage(evals, self._handlers())  # no raise

    def test_tolerates_eval_without_assertions(self) -> None:
        evals = [{"id": "e1", "should_trigger": True}]
        assert_handler_coverage(evals, self._handlers())  # no raise

    def test_raises_listing_every_gap(self) -> None:
        evals = [
            {"id": "e1", "assertions": [{"id": "a"}, {"id": "x"}]},
            {"id": "e2", "assertions": [{"id": "y"}]},
        ]
        with pytest.raises(KeyError) as exc:
            assert_handler_coverage(evals, self._handlers())
        message = str(exc.value)
        assert "e1::x" in message
        assert "e2::y" in message
        assert "e1::a" not in message


class TestLoadEvals:
    """Reading an evals.json, with an optional load-time coverage check."""

    @staticmethod
    def _write(tmp_path: Path, payload: dict) -> Path:
        path = tmp_path / "evals.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def test_returns_evals_without_checking_handlers_when_omitted(
        self, tmp_path: Path
    ) -> None:
        # With no handlers passed, the unhandled assertion is not validated.
        path = self._write(
            tmp_path, {"evals": [{"id": "e1", "assertions": [{"id": "a"}]}]}
        )
        assert load_evals(path) == [
            {"id": "e1", "assertions": [{"id": "a"}]}
        ]

    def test_returns_evals_when_handlers_cover_every_assertion(
        self, tmp_path: Path
    ) -> None:
        path = self._write(
            tmp_path, {"evals": [{"id": "e1", "assertions": [{"id": "a"}]}]}
        )
        evals = load_evals(path, {"a": lambda _r: None})
        assert [ev["id"] for ev in evals] == ["e1"]

    def test_raises_when_an_assertion_lacks_a_handler(
        self, tmp_path: Path
    ) -> None:
        path = self._write(
            tmp_path,
            {"evals": [{"id": "e1", "assertions": [{"id": "missing"}]}]},
        )
        with pytest.raises(KeyError, match="e1::missing"):
            load_evals(path, {"a": lambda _r: None})


class TestEvalChecks:
    def test_collects_assertion_handlers(self) -> None:
        handlers = {"a": lambda _r: None, "b": lambda _r: None}
        item = {"assertions": [{"id": "a"}, {"id": "b"}]}
        assert _eval_checks(item, handlers) == [handlers["a"], handlers["b"]]

    def test_appends_trigger_check_when_should_trigger(self) -> None:
        item = {"assertions": [], "should_trigger": True}
        assert len(_eval_checks(item, {})) == 1

    def test_skips_unregistered_assertion_ids(self) -> None:
        item = {"assertions": [{"id": "missing"}]}
        assert _eval_checks(item, {}) == []


class TestCheckFailures:
    def test_counts_failing_runs(self) -> None:
        assert _check_failures(_runs(True, False, False), _skill_check) == 2

    def test_counts_zero_when_all_pass(self) -> None:
        assert _check_failures(_runs(True, True), _skill_check) == 0


class TestResolveShortfall:
    """The per-check optimistic shortfall that feeds `next_batch_size`.

    Returns the fewest further trials that could settle one undetermined
    check -- the shorter of a clean all-pass streak (clears PASS_THRESHOLD)
    and a clean all-fail streak (drops below FAIL_THRESHOLD) -- falling back
    to the remaining budget when neither resolves in time. Target 2/3.
    """

    def test_takes_pass_route_when_a_passing_streak_settles_sooner(self) -> None:
        # From 5/6, a 3-pass streak clears the bar before any fail streak
        # (which would need 4) could, so the pass route drives the result.
        assert _resolve_shortfall(5, 6, TARGET, 16) == 3

    def test_takes_fail_route_when_a_failing_streak_settles_sooner(self) -> None:
        # From 2/4, a single further failure already drops below the floor,
        # while an all-pass streak would need far more; the fail route wins.
        assert _resolve_shortfall(2, 4, TARGET, 16) == 1

    def test_falls_back_to_remaining_when_neither_streak_settles(self) -> None:
        # From 3/5 with only 2 trials left, neither a 2-pass nor a 2-fail
        # streak clears the band, so it yields the whole remaining budget.
        assert _resolve_shortfall(3, 5, TARGET, 2) == 2


class TestNextBatchSize:
    """The adaptive batch sizing that drives `run_eval_adaptive`.

    A run "passes" `_skill_check` when `skill_invoked` is True. The result is
    0 once the eval verdict is fixed (every check PASS-locked, or any check
    FAIL-locked, or the budget spent), else the optimistic batch -- the
    largest per-undetermined-check shortfall, floored at `BATCH_FLOOR` and
    capped by the remaining budget. Numbers assume target 2/3.
    """

    def test_first_batch_is_the_floor(self) -> None:
        # From scratch the optimistic shortfall (a single failure could FAIL)
        # is below the floor, so the floor drives the opening salvo.
        assert next_batch_size([], [_skill_check], 21, TARGET) == BATCH_FLOOR

    def test_zero_once_pass_locked(self) -> None:
        runs = _runs(*([True] * 6))  # p_good > PASS_THRESHOLD
        assert next_batch_size(runs, [_skill_check], 21, TARGET) == 0

    def test_zero_once_fail_locked(self) -> None:
        runs = _runs(False, False, False)  # p_good < FAIL_THRESHOLD
        assert next_batch_size(runs, [_skill_check], 21, TARGET) == 0

    def test_zero_when_budget_exhausted(self) -> None:
        runs = _runs(*([True] * 13 + [False] * 8))  # 21 runs, still undetermined
        assert next_batch_size(runs, [_skill_check], 21, TARGET) == 0

    def test_no_checks_runs_nothing(self) -> None:
        assert next_batch_size(_runs(False, False), [], 21, TARGET) == 0

    def test_uses_shortfall_when_it_exceeds_the_floor(self) -> None:
        # 4 of 5 passes: undetermined, and the optimistic shortfall is 4 (> floor).
        runs = _runs(True, True, True, True, False)
        assert next_batch_size(runs, [_skill_check], 21, TARGET) == 4

    def test_caps_batch_at_remaining_budget(self) -> None:
        # 20 runs done, still undetermined: only 1 trial of budget remains.
        runs = _runs(*([True] * 13 + [False] * 7))
        assert next_batch_size(runs, [_skill_check], 21, TARGET) == 1

    def test_takes_largest_shortfall_across_checks(self) -> None:
        def check_text(run: EvalRun) -> None:
            if not run.assistant_text:
                raise AssertionFailure("text")

        # skill_invoked: 4/5 (shortfall 4); assistant_text: 3/5 (shortfall 2).
        # Both undetermined; the larger shortfall drives the batch.
        runs = [
            EvalRun(eval_id="t", prompt="", skill_invoked=inv, assistant_text=t)
            for inv, t in zip(
                [True, True, True, True, False], ["x", "x", "x", "", ""]
            )
        ]
        assert next_batch_size(runs, [_skill_check, check_text], 21, TARGET) == 4

    def test_fail_locked_check_short_circuits_others(self) -> None:
        def check_text(run: EvalRun) -> None:
            if not run.assistant_text:
                raise AssertionFailure("text")

        # skill_invoked 4/5 (undetermined) but assistant_text 0/5 (FAIL-locked):
        # the eval already fails, so no further trials are run.
        runs = [
            EvalRun(eval_id="t", prompt="", skill_invoked=inv, assistant_text="")
            for inv in [True, True, True, True, False]
        ]
        assert next_batch_size(runs, [_skill_check, check_text], 21, TARGET) == 0


class TestRunEvalAdaptive:
    """The batch loop accumulates runs until `next_batch_size` returns 0."""

    def test_stops_once_pass_locked(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Every scripted trial passes: the opening floor of 3 runs, then a
        # second batch tips the posterior over PASS_THRESHOLD and it stops.
        scripted = [True] * 21
        state = {"i": 0}

        def fake_batch(
            item: dict,
            repo_root: Path,
            skill_name: str,
            count: int,
            *,
            gate: object = None,
            isolate: bool = False,
            model: str,
            runner: object = None,
        ) -> list[EvalRun]:
            chunk = scripted[state["i"] : state["i"] + count]
            state["i"] += count
            return _runs(*chunk)

        monkeypatch.setattr(binom_eval.grading, "run_claude_batch", fake_batch)
        runs = binom_eval.run_eval_adaptive(
            {"id": "t", "prompt": "p"},
            Path("."),
            "demo",
            max_trials=21,
            target=TARGET,
            checks=[_skill_check],
            model="m",
        )
        # First batch is the floor (3); a clean streak then PASS-locks, so it
        # stops well short of the 21-trial budget.
        assert BATCH_FLOOR <= len(runs) < 21
        assert all(run.skill_invoked for run in runs)

    def test_stops_fast_when_failing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # A skill that always misses: the opening floor of 3 FAIL-locks it.
        def fake_batch(
            item: dict,
            repo_root: Path,
            skill_name: str,
            count: int,
            *,
            gate: object = None,
            isolate: bool = False,
            model: str,
            runner: object = None,
        ) -> list[EvalRun]:
            return _runs(*([False] * count))

        monkeypatch.setattr(binom_eval.grading, "run_claude_batch", fake_batch)
        runs = binom_eval.run_eval_adaptive(
            {"id": "t", "prompt": "p"},
            Path("."),
            "demo",
            max_trials=21,
            target=TARGET,
            checks=[_skill_check],
            model="m",
        )
        assert len(runs) == BATCH_FLOOR


class TestTriggerPassCounts:
    def test_counts_invoked_trials(self) -> None:
        runs = {"a": _runs(True, False, True)}
        evals = [{"id": "a", "should_trigger": True}]
        assert trigger_pass_counts(runs, evals) == [("a", 2, 3)]

    def test_ignores_non_should_trigger_evals(self) -> None:
        runs = {"a": _runs(False)}
        evals = [{"id": "a", "should_trigger": False}]
        assert trigger_pass_counts(runs, evals) == []

    def test_returns_empty_when_no_evals(self) -> None:
        assert trigger_pass_counts({}, []) == []


class TestExpandEvals:
    def test_expands_prompt_template_and_fixture(self, tmp_path: Path) -> None:
        fixture_dir = tmp_path / "fixtures"
        fixture_dir.mkdir()
        (fixture_dir / "sample.ts").write_text("export const x = 1;\n")
        evals_path = tmp_path / "evals.json"
        evals_path.write_text(
            json.dumps(
                {
                    "evals": [
                        {
                            "id": "demo",
                            "prompt_template": "Refactor:\n```\n{fixture}\n```\n",
                            "fixture": "fixtures/sample.ts",
                            "assertions": [],
                        }
                    ]
                }
            )
        )
        evals = expand_evals(evals_path)
        assert evals[0]["prompt"] == "Refactor:\n```\nexport const x = 1;\n\n```\n"
        assert evals[0]["prompt_input"] == "export const x = 1;\n"

    def test_load_evals_passes_through_literal_prompt(self, tmp_path: Path) -> None:
        evals_path = tmp_path / "evals.json"
        evals_path.write_text(
            json.dumps({"evals": [{"id": "plain", "prompt": "hello", "assertions": []}]})
        )
        assert load_evals(evals_path)[0]["prompt"] == "hello"

    def test_returns_item_unchanged_without_template_or_fixture(
        self, tmp_path: Path
    ) -> None:
        item = {"id": "plain", "prompt": "hello", "assertions": []}
        assert expand_eval_item(item, tmp_path) == item

    def test_raises_when_only_fixture_is_provided(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="needs both prompt_template and fixture"):
            expand_eval_item(
                {"id": "broken", "fixture": "missing-template.txt", "assertions": []},
                tmp_path,
            )

    def test_raises_when_only_template_is_provided(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match=r"eval 'broken' needs both"):
            expand_eval_item(
                {
                    "id": "broken",
                    "prompt_template": "Use:\n{fixture}",
                    "assertions": [],
                },
                tmp_path,
            )

    def test_expansion_removes_template_and_fixture_keys(self, tmp_path: Path) -> None:
        fixture = tmp_path / "sample.txt"
        fixture.write_text("fixture body\n")
        result = expand_eval_item(
            {
                "id": "demo",
                "prompt_template": "Content:\n{fixture}",
                "fixture": "sample.txt",
                "assertions": [],
            },
            tmp_path,
        )
        assert "prompt_template" not in result
        assert "fixture" not in result
        assert result["prompt"] == "Content:\nfixture body\n"