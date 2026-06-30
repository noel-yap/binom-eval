# Example: grading a skill with binom-eval

`example-skill/evals/` is a complete, self-contained eval suite for a
hypothetical skill named `example-skill`. It shows the four pieces every
consumer supplies:

| File | Role |
| --- | --- |
| `evals.json` | The prompts, per-eval assertion ids, and `should_trigger` flags. |
| `_assertions.py` | `ASSERTION_HANDLERS`: one handler per assertion id, each raising `AssertionFailure` on failure. Built on `binom_eval`'s text helpers. |
| `conftest.py` | Binds the session-scoped `eval_runs` fixture via `bind_eval_runs_fixture(...)`. |
| `test_evals.py` | Registers the standard live-eval tests via `register_live_eval_tests(...)`. |

Run it (needs the `claude` CLI on `PATH`):

```bash
pytest examples/example-skill/evals -m live_eval
```

Without `claude` on `PATH` the `eval_runs` fixture skips, so collection still
succeeds. The `--live-eval-max-trials` / `--live-eval-target-rate` options and
the `live_eval` marker come from the installed `binom_eval` pytest plugin —
no `sys.path` wiring or hook re-exports needed.

## How grading works here

`make_eval_runs_fixture` runs each eval through `run_eval_adaptive`: trials
fire in concurrent batches and re-grade after each, stopping as soon as the
Beta-binomial posterior locks every assertion PASS or any assertion FAIL.
`register_live_eval_tests` turns each assertion's per-trial outcomes into a
verdict with `assert trial_outcomes_passed(...)` and
`trial_outcomes_failure_message(...)` on failure — the posterior must put ≥
½ of its mass at or above the target rate. The `should_trigger` rollup grades
whether the skill actually fired.
