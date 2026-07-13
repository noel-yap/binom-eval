# binom-eval task runner. Targets wrap `uv` so a fresh checkout needs only
# `make test`. Override pytest args with ARGS, e.g. `make test ARGS=-k grading`.
# Choose a release bump with BUMP, e.g. `make release BUMP=minor`.

ARGS ?=
BUMP ?=

.DEFAULT_GOAL := test

.PHONY: help sync test test-all test-live example clean release release-dry

help: ## List available targets
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "} {printf "  %-12s %s\n", $$1, $$2}'

sync: ## Install the package and dev deps into .venv
	uv sync

test: sync ## Run the fast unit suite (no live `claude -p` calls)
	uv run pytest -m 'not live_eval' $(ARGS)

test-all: sync ## Run every test, including live evals (needs `claude` on PATH)
	uv run pytest $(ARGS)

test-live: sync ## Run only the live evals (needs `claude` on PATH)
	uv run pytest -m live_eval $(ARGS)

example: sync ## Run the bundled example eval suite (needs `claude` on PATH)
	uv run pytest examples/example-skill/evals -m live_eval $(ARGS)

release-dry: ## Preview the next release version + notes without tagging
	./scripts/release.sh --dry-run $(BUMP)

release: ## Tag+push a release (infers bump; BUMP=major|minor|patch|X.Y.Z overrides)
	./scripts/release.sh $(BUMP)

clean: ## Remove caches and build artifacts
	rm -rf .pytest_cache build dist *.egg-info
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
