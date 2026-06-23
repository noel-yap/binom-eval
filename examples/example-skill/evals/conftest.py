"""Pytest wiring for the example-skill eval suite.

Run with:

    pytest examples/example-skill/evals -m live_eval
"""

from __future__ import annotations

from pathlib import Path

from binom_eval import bind_eval_runs_fixture

from ._assertions import ASSERTION_HANDLERS

EVAL_DIR = Path(__file__).resolve().parent
SKILL_NAME = EVAL_DIR.parent.name

eval_runs = bind_eval_runs_fixture(EVAL_DIR, SKILL_NAME, ASSERTION_HANDLERS)
