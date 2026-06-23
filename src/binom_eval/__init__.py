"""Bayesian (Beta-binomial) grading for AI skill / agent evals.

`binom_eval` runs an eval suite by invoking `claude -p` against an
`evals.json`, parsing the stream-json output, and grading each assertion
over repeated live runs. Because a model's output is non-deterministic,
every check is treated as a Bernoulli trial with an unknown true pass rate
``theta``; the suite estimates ``theta`` with a Beta-binomial posterior and
decides PASS/FAIL from how much posterior mass clears a target rate, running
only as many trials as the verdict needs. The package is split by concern:

  * `text_utils` -- pure text/regex helpers for assertion modules
    (code-block extraction, the function-definition regexes, substring
    checks).
  * `stream_json` -- the `EvalRun` dataclass and `parse_stream_json`, which
    turn one `claude -p` run's stdout into an `EvalRun`.
  * `runner` -- the subprocess/env layer: `run_claude`, the concurrent
    `run_claude_batch` (throttled by a shared semaphore), and
    `isolated_workdir` for per-run filesystem isolation.
  * `grading` -- the Beta-binomial verdict (`posterior_pass_prob`,
    `eval_passed`), the adaptive trial driver (`next_batch_size`,
    `run_eval_adaptive`), and the rollups used to grade a batch.
  * `plugin` -- the pytest options, the `live_eval` marker,
    `live_eval_target_rate`, and `make_eval_runs_fixture`. Registered as a
    pytest plugin (entry point `pytest11`), so installing the package wires
    up the `--live-eval-*` options and the `live_eval` marker automatically.

This `__init__` re-exports the public surface, so consumers import
`from binom_eval import ...`. A per-eval-suite `conftest.py` binds
`make_eval_runs_fixture(...)` to the name `eval_runs`, supplying its own
skill name, repo root, `evals.json` path, and assertion handlers; see
`examples/` for the consumer pattern.

Evals are inherently non-deterministic, so each is graded over repeated
live runs; there is deliberately no result caching (deterministic tests
belong in a separate unit suite, not here).

Stdlib + pytest only.
"""

from __future__ import annotations

from binom_eval.grading import (
    BATCH_FLOOR,
    FAIL_THRESHOLD,
    PASS_THRESHOLD,
    PRIOR_ALPHA,
    PRIOR_BETA,
    _check_failures,
    _eval_checks,
    _trigger_check,
    assert_eval_passed,
    assert_handler_coverage,
    eval_passed,
    expand_eval_item,
    expand_evals,
    failing_assertions,
    load_evals,
    next_batch_size,
    posterior_pass_prob,
    run_eval_adaptive,
    trial_outcomes,
    trigger_pass_counts,
)
from binom_eval.plugin import (
    DEFAULT_CONCURRENCY,
    DEFAULT_MAX_TRIALS,
    DEFAULT_TARGET_RATE,
    live_eval_target_rate,
    make_eval_runs_fixture,
    pytest_addoption,
    pytest_configure,
)
from binom_eval.runner import (
    DEFAULT_TIMEOUT_SECONDS,
    ISOLATION_IGNORE,
    NESTED_SESSION_MARKERS,
    isolated_workdir,
    run_claude,
    run_claude_batch,
    stripped_env,
)
from binom_eval.stream_json import (
    EvalRun,
    _assistant_content_blocks,
    _content_blocks_from_event,
    _is_assistant_event,
    _is_skill_hit,
    _message_from_event,
    _text_from_block,
    _try_parse_json,
    agent_invoked,
    parse_stream_json,
    skill_invoked_in_tools,
    tool_invoked,
)
from binom_eval.text_utils import (
    ARROW_FN_RE,
    CODE_BLOCK_RE,
    NAMED_FN_RE,
    code_blocks,
    first_line,
    missing_from,
)

__all__ = [
    # text_utils
    "ARROW_FN_RE",
    "CODE_BLOCK_RE",
    "NAMED_FN_RE",
    "code_blocks",
    "first_line",
    "missing_from",
    # stream_json
    "EvalRun",
    "agent_invoked",
    "parse_stream_json",
    "skill_invoked_in_tools",
    "tool_invoked",
    # runner
    "DEFAULT_TIMEOUT_SECONDS",
    "ISOLATION_IGNORE",
    "NESTED_SESSION_MARKERS",
    "isolated_workdir",
    "run_claude",
    "run_claude_batch",
    "stripped_env",
    # grading
    "BATCH_FLOOR",
    "FAIL_THRESHOLD",
    "PASS_THRESHOLD",
    "PRIOR_ALPHA",
    "PRIOR_BETA",
    "assert_eval_passed",
    "assert_handler_coverage",
    "eval_passed",
    "expand_eval_item",
    "expand_evals",
    "failing_assertions",
    "load_evals",
    "next_batch_size",
    "posterior_pass_prob",
    "run_eval_adaptive",
    "trial_outcomes",
    "trigger_pass_counts",
    # plugin
    "DEFAULT_CONCURRENCY",
    "DEFAULT_MAX_TRIALS",
    "DEFAULT_TARGET_RATE",
    "live_eval_target_rate",
    "make_eval_runs_fixture",
    "pytest_addoption",
    "pytest_configure",
]