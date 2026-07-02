# binom-eval

**Bayesian (Beta-binomial) grading for AI skill / agent evals.**

`binom-eval` grades a Claude skill or agent by running it live, repeatedly,
and deciding pass/fail from a posterior over its true success rate — not from
a single run or a brittle count threshold. It's built for evals whose outcome
is genuinely non-deterministic: the same prompt can pass on one run and fail
on the next, so the only honest verdict is a statistical one.

## The idea

Each graded check is a **Bernoulli trial**: on any single `claude -p` run the
skill either satisfies the assertion (with unknown true pass rate `θ`) or it
doesn't. We never observe `θ` — only `k` passes out of `n` trials. So instead
of thresholding a raw count, `binom-eval` puts a posterior on `θ` and asks how
much of it clears a target rate.

- **Model:** `k ~ Binomial(n, θ)`, prior `θ ~ Beta(1, 1)` (uniform). Beta is
  conjugate to the binomial, so the posterior is closed-form:
  `θ | (k, n) ~ Beta(1 + k, 1 + (n − k))` — each batch of trials just bumps
  the two parameters, no sampling.
- **Bar:** a target rate (default `3/5`) is the true pass rate a good skill
  should clear. `posterior_pass_prob` returns `p_good = P(θ ≥ target | k, n)`
  via the regularized incomplete beta function — **stdlib only**, no SciPy.
- **Verdict band:** PASS once `p_good > 1 − e⁻²` (≈ 0.865), FAIL once
  `p_good < e⁻²` (≈ 0.135); in between, the evidence is inconclusive and more
  trials are worth running. The band is symmetric, so an early unlucky streak
  doesn't lock a verdict.
- **Adaptive trials:** trials run in concurrent batches; after each batch the
  posterior is re-graded. Sampling stops as soon as every check locks PASS or
  any check locks FAIL — capping cost at `--live-eval-max-trials` (default 21)
  while usually spending far fewer.
- **Concurrency:** the evals in a suite are driven in parallel, and each fans
  its trial batches out too. A single shared semaphore
  (`--live-eval-concurrency`, default 5) caps total in-flight `claude -p` calls
  across the whole session, so load and API-rate pressure stay bounded no
  matter how large the suite; set it to `1` to run fully serially. Skills that
  **write** to the working tree need `--live-eval-isolate`, which runs each
  trial in a throwaway copy of the repo root so concurrent runs can't clobber
  one another. The shared semaphore lives **inside one process**, so running
  under [pytest-xdist](https://pytest-xdist.readthedocs.io) (`-n N`) gives each
  of the `N` workers its own gate — total in-flight calls become
  `N × --live-eval-concurrency`, and because the eval fixture is
  session-scoped, every worker recomputes the *whole* suite. The built-in
  parallelism already saturates the gate, so the simplest answer is to **not
  pass `-n`** for the eval run. If you must shard across workers anyway, use
  `--dist loadgroup` (or `loadscope`) to keep a skill's eval tests on a single
  worker, and divide `--live-eval-concurrency` by the worker count to hold the
  global ceiling.

Evals are non-deterministic by design and are **never cached**: every trial is
a fresh live `claude -p` call, so the suite measures the model's run-to-run
variance. Deterministic checks belong in an ordinary unit suite.

## Why these defaults

The defaults are tuned for the expected workload: **most eval runs are of
working skills in CI** (a broken skill gets fixed fast, so it's rarely the
thing under test). That makes the dominant failure mode a *false red* — a
working skill that the build rejects by chance — so the parameters are chosen
to keep that rare while still catching real regressions. The numbers below are
from Monte-Carlo simulation of the adaptive loop (budget 21, prior
`Beta(1, 1)`); "false-FAIL" is a good skill wrongly failed, "caught" is a
broken skill correctly failed.

- **`TARGET_RATE = 3/5`.** The bar must sit *below* where good skills actually
  live (~0.9+), because asking the posterior to distinguish 0.90 from a bar
  near it is both expensive and flaky. At 3/5 a true-0.90 skill false-fails
  only ~0.2% of the time (true-0.80: ~3%), while clearly-broken skills are
  still caught reliably (true-0.40 ~91%, true-0.30 ~100%). This deliberately
  favours **never red-flagging a working skill** over catching *mildly*-broken
  ones: a true-0.60 skill is caught only ~46% (vs ~79% at a 2/3 bar), on the
  assumption that real regressions crater well below 0.6 and get fixed fast.
  Raise the bar toward 2/3 if catching mild breakage matters more than CI
  quiet. "Passes at least three of every five attempts" is an easy bar to
  explain, and 0.6 sits just under the golden ratio (~0.618) that the
  Fibonacci-ratio candidates we compared converge to.
- **Band `(e^-2, 1 - e^-2)` ≈ (0.135, 0.865).** Symmetric about ½, so an early
  unlucky streak is as hard to lock a FAIL on as a lucky one is to lock a PASS.
  `e^-2` is a natural "two-units-of-evidence" tail. Raising the low edge (e.g.
  to 0.5, a FAIL-eager asymmetric band) was measured to ~10× the false-FAIL
  rate on good skills — rejected.
- **`BATCH_FLOOR = 3`.** Not just a concurrency knob — it's a *stability* knob.
  Flooring the opening salvo at 3 forces a representative sample before the
  posterior may commit, which cut false-FAIL ~3× versus a floor of 1 (e.g.
  12% → 4% at target 0.7, true 0.9) for ~2 extra trials. A floor of 2 was
  strictly worse (same cost, less benefit, and it could *raise* round counts);
  5 bought marginal speed at near-max trial cost. 3 is the sweet spot.
- **`MAX_TRIALS = 21`.** A ceiling, not a target: good skills lock in ~2–3
  rounds (~6–9 trials) and never approach it. It only bites for a skill
  sitting *exactly* at the bar, which is genuinely undecidable — one more
  trial can't rescue it. 21 = 3 × 7 divides evenly by `BATCH_FLOOR`, so the
  worst case is a clean seven rounds of three with no ragged final batch. The
  budget is the least sensitive parameter here (20 vs 21 was within noise).
- **Prior `Beta(1, 1)` (uniform).** No prior opinion on a skill's pass rate —
  the verdict is driven by the trials, not by a thumb on the scale. Raise
  `PRIOR_ALPHA` for an optimistic prior ("skills usually work, demand less
  evidence") or `PRIOR_BETA` for a skeptical one.
- **Budget tiebreak at `p_good >= 0.5`.** If a run exhausts the budget still
  inside the band, it's graded toward whichever side holds the majority of the
  posterior. This only matters for at-the-bar skills (everything else locks via
  the band first); 0.5 is the principled midpoint.

`TARGET_RATE` and `MAX_TRIALS` are per-run overridable from the CLI
(`--live-eval-target-rate`, `--live-eval-max-trials`); the band, floor, prior,
and tiebreak are module constants in `binom_eval.grading` — change them there
if the workload assumptions shift.

## Install

Not on PyPI yet — install from Git:

```bash
uv add "binom-eval @ git+https://github.com/noel-yap/binom-eval"
# or: pip install "git+https://github.com/noel-yap/binom-eval"
# pin a release: ...binom-eval.git@v0.1.0
```

Installing registers a pytest plugin, so the `--live-eval-max-trials`,
`--live-eval-target-rate`, `--live-eval-concurrency`, `--live-eval-isolate`,
`--live-eval-model`, and `--live-eval-failure-max-chars` options and the
`live_eval` marker become available to your test suite with no extra wiring. Live evals require the `claude` CLI on
`PATH`; when it's absent the fixture skips rather than fails.

## Usage

A skill's eval suite supplies four things and lets `binom-eval` do the rest:

1. an **`evals.json`** — the prompts and per-eval assertion ids (each eval
   may supply a literal `"prompt"` or a `"prompt_template"` + `"fixture"` pair;
   fixture paths are relative to the directory containing `evals.json`);
2. **assertion handlers** — `dict[str, Callable[[EvalRun], None]]`, each
   raising `AssertionFailure` on failure;
3. a **`conftest.py`** that binds the `eval_runs` fixture; and
4. a **`test_evals.py`** that grades the runs.

```python
# conftest.py
from pathlib import Path
from binom_eval import bind_eval_runs_fixture
from ._assertions import ASSERTION_HANDLERS

EVAL_DIR = Path(__file__).resolve().parent
SKILL_NAME = EVAL_DIR.parent.name          # the skill Claude loads

eval_runs = bind_eval_runs_fixture(
    EVAL_DIR, SKILL_NAME, ASSERTION_HANDLERS,
    repo_root=EVAL_DIR.parents[3],         # omit to run in EVAL_DIR
)
```

```python
# test_evals.py
from pathlib import Path
from binom_eval import register_live_eval_tests
from ._assertions import ASSERTION_HANDLERS

EVAL_DIR = Path(__file__).resolve().parent

register_live_eval_tests(
    globals(),
    evals_path=EVAL_DIR / "evals.json",
    handlers=ASSERTION_HANDLERS,
    subject_name=EVAL_DIR.parent.name,
    trigger="skill",                       # or "agent" for agent suites
)
```

Run the live suite:

```bash
pytest path/to/evals -m live_eval
# demand a higher true rate over a smaller budget:
pytest path/to/evals -m live_eval \
    --live-eval-target-rate 0.8 --live-eval-max-trials 12
# run more trials at once; isolate runs for a skill that writes to the tree:
pytest path/to/evals -m live_eval \
    --live-eval-concurrency 8 --live-eval-isolate
# select a specific model (default: haiku):
pytest path/to/evals -m live_eval \
    --live-eval-model claude-sonnet-4-6
```

An unknown model is rejected before any trial runs. For the `claude` backend
the harness queries the Anthropic Models API to validate the model and, when
the model is not found, includes the list of valid models in the error:

```
model not found: claude-nope; valid models: claude-haiku-4-5-20251001, ...
```

If the API is unreachable the harness falls back to a cheap `claude -p` probe
so a transient network hiccup never blocks a run that might otherwise succeed.

See [`examples/`](examples/) for the full consumer pattern, including
`_assertions.py` text helpers and the `should_trigger` skill-invocation check.

If your suites import siblings relatively (`from ._assertions import ...`) so
several skills' eval dirs can be collected in one run, configure pytest for
namespace-package collection:

```toml
# pyproject.toml
[tool.pytest.ini_options]
addopts = "--import-mode=importlib"
consider_namespace_packages = true
```

## Public API

| Symbol | Purpose |
| --- | --- |
| `AssertionFailure` | Structured failure type for assertion handlers (`summary` + optional labeled `sections`). |
| `make_eval_runs_fixture` | Build the session-scoped `eval_runs` pytest fixture. |
| `bind_eval_runs_fixture`, `register_live_eval_tests` | Thin suite wiring for `conftest.py` / `test_evals.py`. |
| `run_eval_adaptive`, `next_batch_size` | The adaptive trial driver. |
| `posterior_pass_prob`, `eval_passed` | Beta-binomial posterior + final grade. |
| `trial_outcomes_passed`, `trial_outcomes_failure_message`, `failing_assertions`, `trigger_pass_counts` | Grading rollups for a completed batch. |
| `run_claude`, `run_claude_batch`, `stripped_env` | The `claude -p` I/O layer. |
| `EvalRun`, `parse_stream_json` | Stream-json parsing into the shared record. |
| `agent_invoked`, `skill_invoked_in_tools`, `skill_was_invoked`, `agent_or_skill_invoked`, `tool_invoked` | Inspect `EvalRun` for Agent/Skill delegation (bool predicates for use with `assert`). |
| `code_blocks`, `contains`, `contains_all`, `has_code_blocks`, `first_line`, `missing_from`, `NAMED_FN_RE`, `ARROW_FN_RE` | Assertion text/regex helpers. |
| `load_evals`, `expand_evals`, `assert_handler_coverage` | Load + expand + validate an `evals.json`. |

## Requirements

- Python ≥ 3.11
- `pytest` (a runtime dependency — the package *is* a pytest plugin)
- the `claude` CLI on `PATH` for live evals (unit tests need neither)

## Development

```bash
make test        # fast unit suite (no live `claude -p` calls)
make test-all    # every test, including live evals (needs `claude` on PATH)
make help        # list all targets
```

`make` wraps `uv`; a fresh checkout needs only `make test`. Override pytest
args with `ARGS`, e.g. `make test ARGS="-k grading"`. The equivalent without
make is `uv sync && uv run pytest -m 'not live_eval'`.
