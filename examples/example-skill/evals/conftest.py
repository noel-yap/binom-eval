"""Pytest wiring for the example-skill eval suite.

The `--live-eval-*` options, the `live_eval` marker, and the
`live_eval_target_rate` fixture are registered by the installed `binom_eval`
pytest plugin -- this file only binds the session-scoped `eval_runs` fixture
for this skill.

Run with:

    pytest examples/example-skill/evals -m live_eval
"""

from __future__ import annotations

from pathlib import Path

from binom_eval import make_eval_runs_fixture

from ._assertions import ASSERTION_HANDLERS

EVAL_DIR = Path(__file__).resolve().parent
EVALS_PATH = EVAL_DIR / "evals.json"
# The skill directory (the parent of `evals/`) names the skill Claude loads
# and is what the skill-invocation check matches against.
SKILL_NAME = EVAL_DIR.parent.name
# Where `claude -p` runs. For this example the suite dir is fine; a real skill
# would point this at the repo whose files the prompts reference.
REPO_ROOT = EVAL_DIR

eval_runs = make_eval_runs_fixture(
    EVALS_PATH, REPO_ROOT, SKILL_NAME, ASSERTION_HANDLERS
)
