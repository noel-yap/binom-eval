# binom-eval task runner. Targets wrap `uv` so a fresh checkout needs only
# `make test`. Override pytest args with ARGS, e.g. `make test ARGS=-k grading`.
# Choose a release bump with BUMP, e.g. `make release BUMP=minor`.

ARGS ?=
BUMP ?=
BASE ?= origin/main
BASE ?= origin/main

.DEFAULT_GOAL := test

.PHONY: help sync test test-all test-live coverage coverage-compare example clean release release-dry

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

coverage: sync ## Run the fast unit suite and report coverage (terminal + htmlcov/)
	uv run coverage run --source=binom_eval -m pytest -m 'not live_eval' $(ARGS)
	uv run coverage report --show-missing
	uv run coverage html

coverage-compare: sync ## Fail if coverage of files changed vs BASE (default origin/main) decreased
	rm -rf $(CURDIR)/.cov-base && git worktree prune
	git worktree add --detach $(CURDIR)/.cov-base $(BASE)
	cd .cov-base && uv run --with coverage coverage run --source=binom_eval -m pytest -m 'not live_eval' -q \
		&& uv run --with coverage coverage json -o ../.coverage-base.json
	uv run coverage run --source=binom_eval -m pytest -m 'not live_eval' -q
	uv run coverage json -o .coverage-head.json
	python3 scripts/coverage_compare.py .coverage-base.json .coverage-head.json \
		$$(git diff --name-only --diff-filter=AM $(BASE)...HEAD -- src)
	git worktree remove --force .cov-base

coverage-compare: sync ## Fail if coverage of src files changed vs BASE (default origin/main) decreased
	@files=$$(git diff --name-only --diff-filter=AM $(BASE)...HEAD -- 'src/*.py' | paste -sd, -); \
	if [ -z "$$files" ]; then echo "No changed src files."; exit 0; fi; \
	set -e; rm -rf .cov-base; git worktree prune; \
	git worktree add --detach .cov-base $(BASE); \
	trap 'git worktree remove --force .cov-base' EXIT; \
	(cd .cov-base && uv run --with coverage coverage run --include="$$files" -m pytest -m 'not live_eval' -q \
		&& uv run --with coverage coverage json -o ../.coverage-base.json); \
	uv run coverage run --include="$$files" -m pytest -m 'not live_eval' -q; \
	uv run coverage json -o .coverage-head.json; \
	python3 scripts/coverage_compare.py .coverage-base.json .coverage-head.json $$(echo "$$files" | tr , ' ')

example: sync ## Run the bundled example eval suite (needs `claude` on PATH)
	uv run pytest examples/example-skill/evals -m live_eval $(ARGS)

release-dry: ## Preview the next release version + notes without tagging
	./scripts/release.sh --dry-run $(BUMP)

release: ## Tag+push a release (infers bump; BUMP=major|minor|patch|X.Y.Z overrides)
	./scripts/release.sh $(BUMP)

clean: ## Remove caches and build artifacts
	rm -rf .pytest_cache .coverage .coverage-base.json .coverage-head.json htmlcov build dist *.egg-info
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
