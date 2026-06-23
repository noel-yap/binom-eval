"""End-to-end Claude evals for the example-skill."""

from __future__ import annotations

from pathlib import Path

from binom_eval import register_live_eval_tests

from ._assertions import ASSERTION_HANDLERS

EVAL_DIR = Path(__file__).resolve().parent

register_live_eval_tests(
    globals(),
    evals_path=EVAL_DIR / "evals.json",
    handlers=ASSERTION_HANDLERS,
    subject_name=EVAL_DIR.parent.name,
    trigger="skill",
)
