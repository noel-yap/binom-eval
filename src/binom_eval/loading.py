from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from binom_eval.stream_json import EvalRun
from binom_eval.text_utils import BEFORE_AFTER_PROMPT_INSTRUCTION


def assert_handler_coverage(
    evals: list[dict[str, Any]],
    handlers: dict[str, Callable[[EvalRun], None]],
) -> None:
    """Verify every assertion across `evals` has a registered handler.

    Raises:
        KeyError: naming every ``eval_id::assertion_id`` whose `id` is not in
            `handlers`. An assertion with no handler can never be graded, so
            this is a misconfigured suite -- failing once, eagerly, with the
            full list beats discovering each gap later during grading.
    """
    missing = [
        f"{ev['id']}::{assertion['id']}"
        for ev in evals
        for assertion in ev.get("assertions", [])
        if assertion["id"] not in handlers
    ]
    if missing:
        raise KeyError(
            "no handler registered for assertion(s): "
            + ", ".join(missing)
            + "; add them to ASSERTION_HANDLERS"
        )


def expand_eval_item(item: dict[str, Any], eval_dir: Path) -> dict[str, Any]:
    """Return one eval dict with ``prompt_template`` + ``fixture`` expanded.

    The framework injects two prompt additions during expansion. For
    ``should_trigger`` evals the prompt is extended with a constraint
    naming the only skill that should be used; the skill name is derived
    from the directory layout: ``evals.json`` sits two levels below the
    skill root (``{skill}/evals/{lang}/evals.json``), so
    ``eval_dir.parents[1].name`` is the skill name without needing a
    redundant field in ``evals.json``. Every expanded eval -- trigger and
    non-trigger alike -- then gets ``BEFORE_AFTER_PROMPT_INSTRUCTION``
    appended, telling the model that IF it presents before-and-after
    snippets it must delimit them with the framework's marker lines; the
    instruction's own wording is conditional, so it is harmless when no
    refactor is shown. Evals supplying a finished ``prompt`` (no
    template) are returned unmodified.
    """
    expanded = dict(item)
    fixture = expanded.pop("fixture", None)
    template = expanded.pop("prompt_template", None)
    if fixture is not None and template is not None:
        content = (eval_dir / fixture).read_text(encoding="utf-8")
        if expanded.get("should_trigger"):
            skill_name = eval_dir.parents[1].name
            template += (
                f"\n\nUse only the `{skill_name}` skill."
                " Do not invoke any other skill."
            )
        # Appended pre-format like the trigger constraint; the instruction
        # contains no `{`/`}`, so `.format` passes it through untouched.
        template += "\n\n" + BEFORE_AFTER_PROMPT_INSTRUCTION
        expanded["prompt"] = template.format(fixture=content)
        expanded["prompt_input"] = content
    elif fixture is not None or template is not None:
        raise ValueError(
            f"eval {expanded.get('id')!r} needs both prompt_template and fixture"
        )
    return expanded


def expand_evals(evals_path: Path) -> list[dict[str, Any]]:
    """Load ``evals.json`` and expand any ``prompt_template`` + ``fixture`` pairs."""
    eval_dir = evals_path.parent
    raw = json.loads(evals_path.read_text(encoding="utf-8"))["evals"]
    return [expand_eval_item(item, eval_dir) for item in raw]


def load_evals(
    evals_path: Path,
    handlers: dict[str, Callable[[EvalRun], None]] | None = None,
) -> list[dict[str, Any]]:
    """Read an `evals.json` file and return its list of eval items.

    Eval entries may supply a finished ``prompt`` or a ``prompt_template`` plus
    ``fixture`` path (relative to the directory containing ``evals.json``).

    Args:
        evals_path: Path to a skill's `evals.json`, an object with an
            `"evals"` key.
        handlers: when given, every assertion id across the loaded evals must
            have a registered handler; otherwise a single `KeyError` names all
            the gaps. This catches a misconfigured suite at load time so
            downstream grading never meets an ungradable assertion.

    Returns:
        The value of the file's top-level `"evals"` array, with prompts expanded.
    """
    evals = expand_evals(evals_path)
    if handlers is not None:
        assert_handler_coverage(evals, handlers)
    return evals
