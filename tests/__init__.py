"""Unit tests for the skill-independent `binom_eval` package.

One test module per package submodule (`test_stream_json`, `test_grading`,
`test_text_utils`, `test_plugin`), with the `runner` package mirrored by the
`runner/` test subpackage (`test_runner`, `test_claude_runner`). These cover
the harness logic once, so per-skill `evals/test_helpers.py` files only test
their thin SKILL_NAME-bound wrappers.
"""