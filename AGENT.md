# AGENT.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
make test                          # fast unit suite (no live claude -p calls)
make test-all                      # all tests, including live evals (needs claude on PATH)
make test-live                     # live evals only
make coverage                      # fast unit suite + per-file coverage report (htmlcov/)
make coverage-compare              # fail if files changed vs origin/main lost coverage (BASE=ref)
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
    --live-eval-batch-max 3 \
    --live-eval-concurrency 8 \
    --live-eval-isolate \
    --live-eval-isolate-skill \
    --live-eval-isolate-skill-path=.claude/skills/helper \
    --live-eval-isolate-skill-path=PATH \
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

`--live-eval-isolate-skill` and `--live-eval-isolate-skill-path=PATH` each
imply `--live-eval-isolate` but copy only the skills they name, excluding
all other skills (directories with a `SKILL.md`) in the standard containers
(`SKILL_ROOTS`); shared non-skill entries there (README, `_shared/`) are kept.
For a `PATH` outside those containers only the standard containers' sibling
skills are dropped; siblings in its non-standard parent are NOT excluded. A
symlinked container or skill directory must resolve inside the repo and is
copied as a real directory, and the in-repo target of a symlinked container
(`.claude/skills -> ../shared/skills`) is filtered too, so no sibling skill is
visible at any location. The flag finds the skill by name at
`<root>/{.claude,.cursor}/skills/<skill>/` (or `skills/<skill>/`), ignoring
locations without a `SKILL.md` (an error only if none has one); `PATH` names
a directory explicitly and may be repeated (to keep helper skills the
evaluated skill invokes), relative to the repo root (absolute paths and paths
escaping the root are rejected; under a standard container it must be exactly
`<container>/<name>` holding a `SKILL.md`, never a path inside a skill or the
container itself). The two are additive: the kept skills are the evaluated
skill (with the flag) plus every `PATH`, de-duplicated by location, e.g.
`--live-eval-isolate-skill --live-eval-isolate-skill-path=.claude/skills/helper`
(either option alone works too). Every directory is resolved and validated
once, up front, in `make_eval_runs_fixture` (`resolve_skill_dirs`); a missing
or invalid one fails the run, naming it, before any trial. Neither option (`None`) leaves isolation copying the whole tree;
`isolate_skill` is forwarded to runners/batch runners only when set, so
older custom ones without that parameter keep working.

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

Releases are automatic: every push to `main` that has commits since the last
tag cuts the next version and publishes it. You do not run anything by hand.

```
push to main
  └─ release.yml / tag      scripts/release.sh --ci: infer bump, create + push vX.Y.Z
       └─ build             checkout the new tag, uv build, verify tag == wheel version
            └─ github-release   publish the GitHub Release (auto-generated notes)
```

- **Bump rule** (`scripts/release.sh`, the only implementation): a `type!:` subject
  or `BREAKING CHANGE:` footer → major (minor while pre-1.0); any `feat` → minor;
  otherwise → patch. Major stays 0 while pre-1.0.
- **Nothing to release:** a push with no commits since the last tag (for example a
  re-run of an already-released commit) skips cleanly. Every other push releases,
  including `chore`/`docs`/`ci`/`test` commits, which get a patch bump.
- **Concurrency:** releases are serialized by a workflow-level `concurrency`
  group. Queued runs are not cancelled, so a release in progress is never cut off.
- **Why the publish jobs run in the same run:** a tag pushed with `GITHUB_TOKEN`
  does not start other workflows, so `build` and `github-release` run against the
  tag inside the run that created it.
- **Hand-cut tags still work:** pushing a `vX.Y.Z` tag yourself (e.g.
  `git push origin v0.1.0`) skips `tag` and runs `build` and `github-release`
  against that tag. The `tag` job only runs for branch pushes, and a tag already on
  `HEAD` makes it skip, so one commit is never released twice.
- **Manual run:** `make release-dry` previews the next version without tagging.
  `make release` (or `make release BUMP=minor|major|patch|X.Y.Z`) cuts a release
  from your machine. It refuses to run unless the tree is clean and `main` is in
  sync with the remote. Use it only for an out-of-band version, since main pushes
  already release.

> **Invariant:** each run uses `release.yml` **as it exists at the commit that
> triggered it** (the pushed commit on `main`, or the tagged commit for a
> hand-cut tag). A change to the release workflow therefore takes effect with the
> push that merges it, and a hand-cut tag keeps the workflow it points at. If the
> automation fails after the tag is pushed, the tag is still valid. Build locally
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
| `runner/` | subprocess layer: the `Runner` backends (`ClaudeRunner`, `CursorRunner`) selected by `resolve_runner` from a `backend:model` spec, the throttled backend-agnostic `run_eval_batch` (shared `threading.Semaphore`), per-backend `preflight`/`validate_model`, and `isolated_workdir` (with the pre-resolved `isolate_skill` dirs from `resolve_skill_dirs` narrowing the copy to the kept skills, the evaluated one and/or those named by the repeatable `--live-eval-isolate-skill-path`); `retry.py` holds `RetryPolicy`/`RetryableError` (deadline-aware back-off loop used for the Models API lookup and, via `TRIAL_RETRY`, for transient trial failures — a trial that still errors after retries is returned with `EvalRun.errored=True`) |
| `stream_json.py` | `EvalRun` dataclass, `parse_stream_json` (parses `claude -p` stdout), skill/agent invocation predicates |
| `text_utils.py` | Pure text/regex helpers for assertion modules |

### Verdict logic (posterior.py)

- Prior: `Beta(1, 1)` (uniform). Posterior after `k` passes / `n` trials: `Beta(1+k, 1+(n-k))`.
- `p_good = P(θ ≥ TARGET_RATE | k, n)` via `_betainc` (Lentz continued fraction, stdlib only).
- PASS once `p_good > PASS_THRESHOLD` (≈0.865), FAIL once `p_good < FAIL_THRESHOLD` (≈0.135);
  the PASS edge is overridable per run via `--live-eval-pass-threshold`.
- Adaptive: `next_batch_size` (`batch_max`, from `--live-eval-batch-max`, default `BATCH_MAX`) computes the optimistic shortfall per undetermined check, takes the largest shortfall, capped at `batch_max` (3 by default, no floor) and the remaining budget.
- No separate tiebreak: `eval_passed` grades on the same `pass_threshold` as
  the verdict band's PASS edge, so a check that exhausts `MAX_TRIALS` still
  inside the band grades FAIL.
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
