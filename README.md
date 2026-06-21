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

Evals are non-deterministic by design and are **never cached**: every trial is
a fresh live `claude -p` call, so the suite measures the model's run-to-run
variance. Deterministic checks belong in an ordinary unit suite.

## Install

Not on PyPI yet — install from Git:

```bash
uv add "binom-eval @ git+https://github.com/noelyap/binom-eval"
# or: pip install "git+https://github.com/noelyap/binom-eval"
# pin a release: ...binom-eval.git@v0.1.0
```

Installing registers a pytest plugin, so the `--live-eval-max-trials` /
`--live-eval-target-rate` options and the `live_eval` marker become available
to your test suite with no extra wiring. Live evals require the `claude` CLI
on `PATH`; when it's absent the fixture skips rather than fails.

## Usage

A skill's eval suite supplies four things and lets `binom-eval` do the rest:

1. an **`evals.json`** — the prompts and per-eval assertion ids;
2. **assertion handlers** — `dict[str, Callable[[EvalRun], None]]`, each
   raising `AssertionError` when the run fails that assertion;
3. a **`conftest.py`** that binds the `eval_runs` fixture; and
4. a **`test_evals.py`** that grades the runs.

```python
# conftest.py
from pathlib import Path
from binom_eval import make_eval_runs_fixture
from ._assertions import ASSERTION_HANDLERS

EVAL_DIR = Path(__file__).resolve().parent
SKILL_NAME = EVAL_DIR.parent.name          # the skill Claude loads
REPO_ROOT = EVAL_DIR.parents[3]            # where `claude -p` should run

eval_runs = make_eval_runs_fixture(
    EVAL_DIR / "evals.json", REPO_ROOT, SKILL_NAME, ASSERTION_HANDLERS
)
```

```python
# test_evals.py
from binom_eval import assert_eval_passed, failing_assertions, trial_outcomes

def test_assertions_pass(eval_runs, live_eval_target_rate):
    for eval_id, runs in eval_runs.items():
        ...  # grade each assertion's trial outcomes against the target rate
```

Run the live suite:

```bash
pytest path/to/evals -m live_eval
# demand a higher true rate over a smaller budget:
pytest path/to/evals -m live_eval \
    --live-eval-target-rate 0.8 --live-eval-max-trials 12
```

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
| `make_eval_runs_fixture` | Build the session-scoped `eval_runs` pytest fixture. |
| `run_eval_adaptive`, `next_batch_size` | The adaptive trial driver. |
| `posterior_pass_prob`, `eval_passed` | Beta-binomial posterior + final grade. |
| `assert_eval_passed`, `failing_assertions`, `trigger_pass_counts` | Grading rollups for a completed batch. |
| `run_claude`, `run_claude_batch`, `stripped_env` | The `claude -p` I/O layer. |
| `EvalRun`, `parse_stream_json` | Stream-json parsing into the shared record. |
| `code_blocks`, `first_line`, `missing_from`, `NAMED_FN_RE`, `ARROW_FN_RE` | Assertion text/regex helpers. |
| `load_evals`, `assert_handler_coverage` | Load + validate an `evals.json`. |

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
