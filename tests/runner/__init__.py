"""Unit tests for the `binom_eval.runner` package.

`test_runner` covers the backend-agnostic package layer (env scrubbing, the
per-run workdir, the model-probe parser, and the concurrent batch driver);
`test_claude_runner` covers the `claude -p` backend (`ClaudeRunner`).
"""
