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
    --live-eval-min-trials 5 \
    --live-eval-max-trials 12 \
    --live-eval-concurrency 8 \
    --live-eval-isolate \
    --live-eval-model claude:claude-sonnet-4-6 \
    --live-eval-failure-max-chars 10000 \
    --live-eval-pass-threshold 0.95 \
    --live-eval-show-posterior \
    --live-eval-verbose
```

`--live-eval-model` is required for live evals and selects the backend and
model as `backend:model` (known backends: `claude`, `cursor`). The `backend:`
prefix is mandatory — a bare model with no prefix is an error — so each run
targets a named harness. Each pytest run targets a single backend; run pytest
once per backend to grade on both.

`--live-eval-failure-max-chars` caps each failure section rendered in pytest
output (default 2000; zero or negative disables truncation).

`--live-eval-pass-threshold` overrides the verdict band's PASS edge
(default `PASS_THRESHOLD`; must be strictly between 0.5 and 1.0 --
the FAIL edge follows as its complement). `--live-eval-show-posterior`
prints `P(θ ≥ θ₀ | k, n)` and `max θ₀ (pass@τ | k, n)` for each passing
check, for calibrating the target rate and pass threshold.
`--live-eval-verbose` goes further: for each passing check it prints full
per-trial detail -- the posterior summary plus every trial's sections
(assistant reply, tool uses, or the handler's `assert_check` sections).

## Releasing

The package version is **derived from the git tag** via `hatch-vcs`
(`pyproject.toml` declares `dynamic = ["version"]` + `[tool.hatch.version]
source = "vcs"`). There is no version field to edit and no `chore(release)`
bump commit — the tag is the single source of truth.

```bash
make release-dry              # preview the next version + notes, no tag
make release                  # infer the bump from commits, tag, and push
make release BUMP=minor       # force a level (major|minor|patch)
make release BUMP=2.1.0       # or an explicit version
```

`scripts/release.sh` (what these targets run) verifies the tree is clean, on
`main`, and in sync with the remote, then infers the bump from the Conventional
Commits since the last tag (`type!:` / `BREAKING CHANGE:` → major, `feat` →
minor, else patch — major stays 0 while pre-1.0), creates an annotated
`vX.Y.Z` tag, and pushes it to `upstream` (falling back to `origin`). It does
**not** merge anything — land your changes on `main` first.

Pushing the tag triggers `.github/workflows/release.yml`, which builds the
sdist + wheel (version resolved from the tag) and publishes the GitHub Release
with auto-generated notes.

> **Invariant:** a tag-triggered run uses `release.yml` **as it exists at the
> tagged commit**. Any change to the release workflow must be merged to `main`
> *before* you cut the next tag, or that release runs the old workflow. If a
> release's automation fails, the git tag is still valid — build locally
> (`uv build`) and publish by hand with
> `gh release create vX.Y.Z --generate-notes dist/*`.

## Architecture

`binom-eval` is a pytest plugin (`entry_points["pytest11"]`) for grading non-deterministic AI skill/agent evals with a Beta-binomial posterior. It is stdlib + pytest only — no scipy.

### Module responsibilities

| Module | Role |
|---|---|
| `posterior.py` | Beta-binomial math: `posterior_pass_prob`, `eval_passed`, `_verdict`/`Verdict`, `max_target_at_pass_threshold`, and the stdlib `_betainc` (Beta CDF) |
| `assertions.py` | Assertion protocol: `AssertionFailure`, `assert_check`, and single-trial `evaluate_check` |
| `loading.py` | `evals.json` loading/expansion: `load_evals`, `expand_evals`, `assert_handler_coverage` |
| `reporting.py` | Grading rollups + pytest rendering: `trial_outcomes`, `failing_assertions`, `format_posterior_summary`, and the verbose/failure messages |
| `progress.py` | Progress output: `ProgressEvent` dataclass, `ProgressRenderer` protocol, `TtyRenderer`/`PlainRenderer` strategies, `make_renderer` factory (selects TTY vs plain at construction time) |
| `driver.py` | Adaptive trial driver: `run_eval_adaptive`, `next_batch_size`; the batch executor is an injectable `batch_runner` (defaults to `run_eval_batch`); optional `on_progress` renderer for per-batch and per-eval events |
| `grading.py` | Backward-compatible facade re-exporting every name from the five modules above (kept so existing `binom_eval.grading` imports still resolve) |
| `plugin.py` | pytest integration: `--live-eval-*` CLI options, `live_eval` marker, `make_eval_runs_fixture` |
| `suite.py` | Thin consumer wiring: `bind_eval_runs_fixture` (for `conftest.py`) and `register_live_eval_tests` (for `test_evals.py`) |
| `runner/` | subprocess layer: the `Runner` backends (`ClaudeRunner`, `CursorRunner`) selected by `resolve_runner` from a `backend:model` spec, the throttled backend-agnostic `run_eval_batch` (shared `threading.Semaphore`), per-backend `preflight`/`validate_model`, and `isolated_workdir`; `retry.py` holds `RetryPolicy`/`RetryableError` (deadline-aware back-off loop used for the Models API lookup and, via `TRIAL_RETRY`, for transient trial failures — a trial that still errors after retries is returned with `EvalRun.errored=True`) |
| `stream_json.py` | `EvalRun` dataclass, `parse_stream_json` (parses `claude -p` stdout), skill/agent invocation predicates |
| `text_utils.py` | Pure text/regex helpers for assertion modules |

### Verdict logic (posterior.py)

- Prior: `Beta(1, 1)` (uniform). Posterior after `k` passes / `n` trials: `Beta(1+k, 1+(n-k))`.
- `p_good = P(θ ≥ TARGET_RATE | k, n)` via `_betainc` (Lentz continued fraction, stdlib only).
- PASS once `p_good > PASS_THRESHOLD` (≈0.865), FAIL once `p_good < FAIL_THRESHOLD` (≈0.135);
  the PASS edge is overridable per run via `--live-eval-pass-threshold`.
- Adaptive: `next_batch_size` computes the optimistic shortfall per undetermined check, floors at `BATCH_FLOOR=3`, caps at remaining budget.
- Budget tiebreak at `p_good >= 0.5` if `MAX_TRIALS` exhausted inside the band.
- Errored trials (`EvalRun.errored`: CLI died, `is_error` result event, retries exhausted — see `TRIAL_RETRY` in `runner/`) are excluded from every posterior count via `graded_runs`, but still spend the trial budget.

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
