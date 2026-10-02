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
  doesn't lock a verdict. The **final grade** (`eval_passed`) uses this same
  bar — see [Worked example](#worked-example) for what that means once the
  trial budget runs out.
- **Adaptive trials:** trials run in concurrent batches; after each batch the
  posterior is re-graded. Sampling stops as soon as every check locks PASS or
  any check locks FAIL — capping cost at `--live-eval-max-trials` (default 21)
  while usually spending far fewer. `--live-eval-min-trials` (default 0) can
  force a minimum sample before early stopping is allowed.
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

## Worked example

Say a skill's target is the default `0.6` (three-of-five) and the opening
batch is `BATCH_FLOOR = 3` trials. Here's how a few different outcomes grade,
using the actual posterior math (`posterior_pass_prob`):

| Outcomes so far | `k` (passes) | `n` (trials) | `p_good = P(θ ≥ 0.6 ⎮ k, n)` | Verdict |
| --- | --- | --- | --- | --- |
| ✅ ✅ ✅ | 3 | 3 | **0.870** | **PASS** — clears `PASS_THRESHOLD` (0.8647); stop here |
| ✅ ✅ ❌ | 2 | 3 | 0.525 | UNDETERMINED — inside the band; run another batch |
| ❌ ❌ | 0 | 2 | **0.064** | **FAIL** — drops below `FAIL_THRESHOLD` (0.1353); stop here |
| ✅ × 6 | 6 | 6 | 0.972 | **PASS**, and even more confidently than the 3/3 row |

The first row is why "good skills lock in ~2–3 rounds": three clean passes
against a `0.6` target is already enough evidence, so a healthy skill often
never runs past the opening batch. The third row shows the same on the FAIL
side — two clean misses locks the verdict just as fast, so a badly broken
skill doesn't burn the full budget either. The second row is the case that
keeps sampling: one miss in three trials leaves `p_good` near the middle, so
`next_batch_size` computes another optimistic batch and the loop continues.

**What if the budget runs out and the eval is *still* inside the band?** That
only happens for a skill sitting almost exactly on the target rate. At the
default target and `--live-eval-max-trials 21`:

| `k` (of 21) | `p_good` | In band? | Grade |
| --- | --- | --- | --- |
| 14 | 0.710 | yes | **FAIL** |
| 15 | 0.842 | yes | **FAIL** |
| 16 | 0.928 | no — clears `PASS_THRESHOLD` | **PASS** (would already have locked before trial 21) |

`eval_passed` grades on the *same* `pass_threshold` the band uses
(`p_good > pass_threshold`), not a looser `p_good >= 0.5` — so `k=14` and
`k=15` both grade FAIL even though more than half the posterior favors the
skill. There is no separate budget-exhausted tiebreak: reaching `MAX_TRIALS`
still inside the band is itself close enough to a coin flip that it should
grade the same way a genuinely undecidable check would.

`format_posterior_summary` (surfaced via `--live-eval-show-posterior` /
`--live-eval-verbose`) also reports `max θ₀ (pass@τ | k, n)` — the highest
target rate that *would* still lock PASS given the same `(k, n)`. For the
`k=3, n=3` row above that's `≈0.607`: three clean passes locks a `0.6` target
with a bit of room to spare, but wouldn't yet lock a much higher one like
`0.65`.

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
  rounds (~6–9 trials) and never approach it (see the worked example above).
  It only bites for a skill sitting *exactly* at the bar, which is genuinely
  undecidable — one more trial can't rescue it. 21 = 3 × 7 divides evenly by
  `BATCH_FLOOR`, so the worst case is a clean seven rounds of three with no
  ragged final batch. The budget is the least sensitive parameter here (20 vs
  21 was within noise).
- **Prior `Beta(1, 1)` (uniform).** No prior opinion on a skill's pass rate —
  the verdict is driven by the trials, not by a thumb on the scale. Raise
  `PRIOR_ALPHA` for an optimistic prior ("skills usually work, demand less
  evidence") or `PRIOR_BETA` for a skeptical one.
- **No separate budget-exhausted tiebreak.** If a run exhausts the trial
  budget while still inside the band, it grades on the same `pass_threshold`
  the band's PASS edge uses — there's no looser `p_good >= 0.5` fallback. This
  only matters for skills sitting right at the bar (everything else locks via
  the band first); see the `k=14`/`k=15`/`k=16` table in
  [Worked example](#worked-example) for exactly where that bites.

`TARGET_RATE`, the trial budget, and the band's pass edge are per-run
overridable from the CLI: `--live-eval-target-rate`;
`--live-eval-pass-threshold` (the FAIL edge follows as its complement,
keeping the band symmetric about 1/2; values must sit strictly between
0.5 and 1.0 — and the same value gates the final grade, per above); and
`--live-eval-min-trials` / `--live-eval-max-trials` (`min-trials` must not
exceed `max-trials`). The batch floor (`BATCH_FLOOR`) and prior
(`PRIOR_ALPHA`, `PRIOR_BETA`) are module constants, importable from
`binom_eval` — change them there if the workload assumptions shift.

## Tuning knobs

Every `--live-eval-*` pytest option, its default, and what it actually
changes:

| Flag | Default | What it controls |
| --- | --- | --- |
| `--live-eval-target-rate` | `0.6` (3/5) | The true pass rate `θ` a good skill should clear. Raise it (e.g. `0.8`) to demand a more reliable skill; lower it to tolerate a flakier one. See `TARGET_RATE` above. |
| `--live-eval-pass-threshold` | `≈0.8647` (`1 − e⁻²`) | The verdict band's PASS edge; `p_good` must exceed this to lock PASS, and drop below its complement (`≈0.1353`) to lock FAIL — always symmetric about `0.5`. Must sit strictly between `0.5` and `1.0`. Also the bar `eval_passed` uses for the final grade. Raise it (e.g. `0.95`) to demand more confidence before locking either verdict, at the cost of more trials. |
| `--live-eval-min-trials` | `0` | Floor beneath the adaptive driver's early stop: even once the verdict locks, at least this many trials still run. Must not exceed `--live-eval-max-trials`. |
| `--live-eval-max-trials` | `21` | Budget ceiling: the most times any single eval runs before the verdict is forced. Good skills lock in far fewer trials (see [Worked example](#worked-example)); this only bites a skill sitting almost exactly at the target rate. |
| `--live-eval-concurrency` | `5` | Max backend calls (`claude -p` or otherwise) in flight at once, across the whole session — shared by every eval and every eval's own trial batches. Set to `1` to run fully serially. |
| `--live-eval-isolate` | off | Runs each trial in a throwaway copy of the repo root instead of the shared tree. Needed for skills that **write** to the working tree, so concurrent trials can't clobber each other. |
| `--live-eval-isolate-skill` | off | Implies `--live-eval-isolate`, but the throwaway copy keeps only the skill being evaluated (plus any named by `--live-eval-isolate-skill-path`): other skills (directories with a `SKILL.md`) in the skill containers are excluded, while shared non-skill entries there (a `README.md`, a `_shared/` directory) are kept. A symlinked container or skill directory must resolve inside the repo and is copied as a real directory; the in-repo target of a symlinked container (e.g. `shared/skills` behind `.claude/skills -> ../shared/skills`) is filtered too, so no sibling skill is visible at any location. The skill is found by name at `<repo_root>/.claude/skills/<skill>/` (or the `.cursor/skills/` / `skills/` equivalent); locations without a `SKILL.md` are ignored. Errors up front if no location holds a `SKILL.md`. |
| `--live-eval-isolate-skill-path` | *(unset)* | `PATH`, relative to the repo root; **repeatable**. Like `--live-eval-isolate-skill` (implies `--live-eval-isolate`) but names a skill directory explicitly, so helper skills the evaluated skill invokes can be kept too: `--live-eval-isolate-skill --live-eval-isolate-skill-path=.claude/skills/helper`. Additive with `--live-eval-isolate-skill`: the kept skills are the evaluated skill (if that flag is given) plus every `PATH`, de-duplicated; either option works alone. Other skills (directories with a `SKILL.md`) in the standard containers are dropped, but non-skill entries there are kept; for a path outside them (e.g. `plugins/x/skills/foo`) only those containers' sibling skills are dropped, and siblings in the path's own non-standard parent are NOT excluded. Absolute paths and paths escaping the repo root are rejected, as is a path at or under a standard container unless it is exactly `<container>/<name>` holding a `SKILL.md` (e.g. `.claude/skills/foo/sub` and `.claude/skills` are errors). Any invalid `PATH` fails the run up front, naming it. |
| `--live-eval-model` | *(required — no default)* | `backend:model`, e.g. `claude:claude-haiku-4-5-20251001` or `cursor:sonnet-4.5`. The `backend:` prefix is mandatory (known backends: `claude`, `cursor`), so every run names a single, explicit harness. |
| `--live-eval-timeout` | `300` (seconds) | Per-trial subprocess deadline; all retry attempts for one trial share this budget, so wall time per trial is capped at this value. |
| `--live-eval-failure-max-chars` | `2000` | Per-section character cap when a trial's structured sections (assistant reply, tool uses, assertion sections) are rendered — for failures always, and for passes too under `--live-eval-verbose`. Zero or negative disables truncation. |
| `--live-eval-show-posterior` | off | After each passing check, print the one-line posterior summary: `P(θ ≥ θ₀ ⎮ k, n)` and `max θ₀ (pass@τ ⎮ k, n)`. |
| `--live-eval-verbose` | off | Everything `--live-eval-show-posterior` prints, plus every trial's full detail (assistant reply, tool uses, or the handler's `assert_check` sections) for each passing check. |
| `--live-eval-progress` | off | Print per-batch progress to stderr while the suite runs (carriage-return overwrite on a TTY, plain newlines in CI/non-TTY environments). |

## Install

Not on PyPI yet — install from Git:

```bash
uv add "binom-eval @ git+https://github.com/noel-yap/binom-eval"
# or: pip install "git+https://github.com/noel-yap/binom-eval"
# pin a release: ...binom-eval.git@v0.1.0
```

Installing registers a pytest plugin, so the `--live-eval-target-rate`,
`--live-eval-pass-threshold`, `--live-eval-min-trials`, `--live-eval-max-trials`,
`--live-eval-concurrency`, `--live-eval-isolate`, `--live-eval-isolate-skill`,
`--live-eval-isolate-skill-path`, `--live-eval-model`,
`--live-eval-failure-max-chars`, `--live-eval-show-posterior`,
`--live-eval-verbose`, `--live-eval-timeout`, and `--live-eval-progress`
options and the
`live_eval` marker become available to your test suite with no extra wiring. Live evals require the `claude` CLI on
`PATH`; when it's absent the fixture skips rather than fails.

## Usage

A skill's eval suite supplies four things and lets `binom-eval` do the rest:

1. an **`evals.json`** — the prompts and per-eval assertion ids (each eval
   may supply a literal `"prompt"` or a `"prompt_template"` + `"fixture"` pair;
   fixture paths are relative to the directory containing `evals.json`).
   Expanded prompts also carry the conditional before/after marker
   instruction (`BEFORE_AFTER_PROMPT_INSTRUCTION`), so a model that shows
   original and refactored code delimits them with the framework's
   sentinel markers;
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

Run the live suite. `--live-eval-model` has no default and is **required**
(format `backend:model`; known backends: `claude`, `cursor`) — every example
below includes it:

```bash
pytest path/to/evals -m live_eval --live-eval-model claude:claude-haiku-4-5-20251001
# demand a higher true rate over a smaller budget:
pytest path/to/evals -m live_eval --live-eval-model claude:claude-haiku-4-5-20251001 \
    --live-eval-target-rate 0.8 --live-eval-max-trials 12
# run at least five trials per eval before early stopping is allowed:
pytest path/to/evals -m live_eval --live-eval-model claude:claude-haiku-4-5-20251001 \
    --live-eval-min-trials 5 --live-eval-max-trials 12
# demand more posterior confidence before a verdict locks:
pytest path/to/evals -m live_eval --live-eval-model claude:claude-haiku-4-5-20251001 \
    --live-eval-pass-threshold 0.95
# also print P(θ ≥ θ₀ | k, n) and max θ₀ (pass@τ | k, n) for each passing check:
pytest path/to/evals -m live_eval --live-eval-model claude:claude-haiku-4-5-20251001 \
    --live-eval-show-posterior
# print full grading detail for every trial of each passing check too --
# the posterior summary plus each trial's sections (assistant reply, tool
# uses, or the handler's assert_check sections), in the same layout as
# failure output; respects --live-eval-failure-max-chars:
pytest path/to/evals -m live_eval --live-eval-model claude:claude-haiku-4-5-20251001 \
    --live-eval-verbose
# run more trials at once; isolate runs for a skill that writes to the tree:
pytest path/to/evals -m live_eval --live-eval-model claude:claude-haiku-4-5-20251001 \
    --live-eval-concurrency 8 --live-eval-isolate
# select a different model, or the cursor backend instead:
pytest path/to/evals -m live_eval --live-eval-model claude:claude-sonnet-4-6
pytest path/to/evals -m live_eval --live-eval-model cursor:sonnet-4.5
# tighten the per-trial deadline (default 300 s) for fast models or CI:
pytest path/to/evals -m live_eval --live-eval-model claude:claude-haiku-4-5-20251001 \
    --live-eval-timeout 60
# print per-batch progress to stderr while the suite runs:
pytest path/to/evals -m live_eval --live-eval-model claude:claude-haiku-4-5-20251001 \
    --live-eval-progress
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
| `AssertionFailure`, `TrialFailure` | Structured failure types: `AssertionFailure` is what assertion handlers raise (`summary` + optional labeled `sections`); `TrialFailure` is the captured, renderable detail behind one graded trial. |
| `make_eval_runs_fixture` | Build the session-scoped `eval_runs` pytest fixture. |
| `bind_eval_runs_fixture`, `register_live_eval_tests` | Thin suite wiring for `conftest.py` / `test_evals.py`. |
| `run_eval_adaptive`, `next_batch_size` | The adaptive trial driver. |
| `posterior_pass_prob`, `eval_passed` | The Beta-binomial posterior (`p_good = P(θ ≥ target ⎮ k, n)`) and the final grade (`p_good > pass_threshold`). |
| `format_posterior_summary`, `max_target_at_pass_threshold` | Human-readable posterior summaries, and the highest target rate that would still lock PASS given the same `(k, n)` (see [Worked example](#worked-example)). |
| `trial_outcomes_passed`, `trial_outcomes_failure_message`, `trial_outcomes_posterior_summary`, `trial_outcomes_verbose_message`, `graded_runs_verbose_message`, `failing_assertions`, `trigger_pass_counts` | Grading rollups and pytest-facing messages for a completed batch of runs. |
| `assert_check`, `evaluate_check` | Helpers for writing assertion handlers — raise (or capture) a labeled `AssertionFailure`. |
| `graded_runs` | The trials that count toward the posterior — errored trials (`EvalRun.errored`, e.g. an API failure that survived the runner's bounded retries) are excluded rather than graded as failures. |
| `PASS_THRESHOLD`, `FAIL_THRESHOLD`, `PRIOR_ALPHA`, `PRIOR_BETA`, `BATCH_FLOOR` | The tunable constants behind the verdict band, the prior, and the opening-batch size — see [Why these defaults](#why-these-defaults). |
| `DEFAULT_TARGET_RATE`, `DEFAULT_MAX_TRIALS`, `DEFAULT_MIN_TRIALS`, `DEFAULT_CONCURRENCY`, `FAILURE_SECTION_MAX_CHARS` | The `--live-eval-*` CLI options' default values — see [Tuning knobs](#tuning-knobs). |
| `run_eval_batch`, `stripped_env` | The backend-agnostic subprocess I/O layer. |
| `ClaudeRunner`, `CursorRunner`, `resolve_runner`, `BACKENDS` | The per-backend `Runner` implementations, and the `backend:model` spec resolver used by `--live-eval-model`. |
| `EvalRun`, `parse_stream_json` | Stream-json parsing into the shared record. |
| `agent_invoked`, `skill_invoked_in_tools`, `skill_was_invoked`, `agent_or_skill_invoked`, `tool_invoked` | Inspect `EvalRun` for Agent/Skill delegation (bool predicates for use with `assert`). |
| `code_blocks`, `contains`, `contains_all`, `fenced_blocks`, `has_code_blocks`, `first_line`, `missing_from`, `NAMED_FN_RE`, `ARROW_FN_RE`, `BEGIN_BEFORE_MARKER`, `END_BEFORE_MARKER`, `BEGIN_AFTER_MARKER`, `END_AFTER_MARKER`, `before_after_snippets`, `BEFORE_AFTER_PROMPT_INSTRUCTION` | Assertion text/regex helpers. |
| `load_evals`, `expand_evals`, `assert_handler_coverage` | Load + expand + validate an `evals.json`. |

## Requirements

- Python ≥ 3.11
- `pytest` ≥ 7.0 (a runtime dependency — the package *is* a pytest plugin)
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

To cut a release once your changes are on `main`:

```bash
make release-dry   # preview the next version, no tag
make release       # infer the bump from commits, tag, and push
```

The version is derived from the git tag (`hatch-vcs`) — no manifest to bump.
Pushing the tag publishes the GitHub Release automatically. See the
**Releasing** section in [`AGENT.md`](AGENT.md) for the full flow.
