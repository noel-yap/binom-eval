# AGENT.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
make test                          # fast unit suite (no live claude -p calls)
make test-all                      # all tests, including live evals (needs claude on PATH)
make test-live                     # live evals only
make test ARGS="-k grading"        # run a subset by keyword
make example                       # run the bundled example eval suite
uv run pytest tests/test_grading.py  # equivalent without make
```

Live eval options:
```bash
pytest path/to/evals -m live_eval \
    --live-eval-target-rate 0.8 \
    --live-eval-max-trials 12 \
    --live-eval-concurrency 8 \
    --live-eval-isolate \
    --live-eval-model claude:claude-sonnet-4-6
```

`--live-eval-model` is required for live evals and selects the backend and
model as `backend:model` (known backends: `claude`, `cursor`). The `backend:`
prefix is mandatory — a bare model with no prefix is an error — so each run
targets a named harness. Each pytest run targets a single backend; run pytest
once per backend to grade on both.

## Architecture

`binom-eval` is a pytest plugin (`entry_points["pytest11"]`) for grading non-deterministic AI skill/agent evals with a Beta-binomial posterior. It is stdlib + pytest only — no scipy.

### Module responsibilities

| Module | Role |
|---|---|
| `grading.py` | Beta-binomial math (`posterior_pass_prob`, `eval_passed`), adaptive trial driver (`run_eval_adaptive`, `next_batch_size`), grading rollups |
| `plugin.py` | pytest integration: `--live-eval-*` CLI options, `live_eval` marker, `make_eval_runs_fixture` |
| `suite.py` | Thin consumer wiring: `bind_eval_runs_fixture` (for `conftest.py`) and `register_live_eval_tests` (for `test_evals.py`) |
| `runner/` | subprocess layer: the `Runner` backends (`ClaudeRunner`, `CursorRunner`) selected by `resolve_runner` from a `backend:model` spec, the throttled `run_claude_batch` (shared `threading.Semaphore`), per-backend `preflight`/`validate_model`, and `isolated_workdir` |
| `stream_json.py` | `EvalRun` dataclass, `parse_stream_json` (parses `claude -p` stdout), skill/agent invocation predicates |
| `text_utils.py` | Pure text/regex helpers for assertion modules |

### Verdict logic (grading.py)

- Prior: `Beta(1, 1)` (uniform). Posterior after `k` passes / `n` trials: `Beta(1+k, 1+(n-k))`.
- `p_good = P(θ ≥ TARGET_RATE | k, n)` via `_betainc` (Lentz continued fraction, stdlib only).
- PASS once `p_good > PASS_THRESHOLD` (≈0.865), FAIL once `p_good < FAIL_THRESHOLD` (≈0.135).
- Adaptive: `next_batch_size` computes the optimistic shortfall per undetermined check, floors at `BATCH_FLOOR=3`, caps at remaining budget.
- Budget tiebreak at `p_good >= 0.5` if `MAX_TRIALS` exhausted inside the band.

### Consumer pattern (see `examples/`)

A skill's eval suite supplies:
1. `evals.json` — prompts + assertion ids; supports `prompt_template` + `fixture` expansion.
2. `_assertions.py` — `dict[str, Callable[[EvalRun], None]]` raising `AssertionFailure` on failure (with optional labeled sections).
3. `conftest.py` — calls `bind_eval_runs_fixture(eval_dir, skill_name, ASSERTION_HANDLERS)` and binds the result to `eval_runs`.
4. `test_evals.py` — calls `register_live_eval_tests(globals(), ...)` which injects three test functions: `test_eval_assertion` (per eval×assertion), `test_eval_expectation` (per-eval rollup), and a trigger rollup.

Multi-suite repos need importlib mode in `pyproject.toml`:
```toml
[tool.pytest.ini_options]
addopts = "--import-mode=importlib"
consider_namespace_packages = true
```
