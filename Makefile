.PHONY: sync
sync:
	uv sync --extra cu128 --group dev

.PHONY: sync-cpu
sync-cpu:
	uv sync --extra cpu --group dev

.PHONY: format
format:
	uv run ruff format
	uv run ruff check --fix

.PHONY: lint
lint:
	uv run ruff check src scripts

.PHONY: test
test:
	uv run pytest

.PHONY: test-cpu
test-cpu:
	FORCE_CPU=1 uv run pytest

.PHONY: check
check: lint test-cpu
