PYTHON ?= python3

.PHONY: install dev-install test lint format typecheck check run-help dry-run-example

install:
	rm -rf dist build
	$(PYTHON) -m build
	$(PYTHON) -m pip install --force-reinstall dist/*.whl

dev-install:
	$(PYTHON) -m pip install -e ".[dev]"

test:
	$(PYTHON) -m pytest

lint:
	$(PYTHON) -m ruff check .

format:
	$(PYTHON) -m black --check .

typecheck:
	$(PYTHON) -m mypy src tests

check: lint format typecheck test

run-help:
	$(PYTHON) -m ollama_zit_import --help

dry-run-example:
	$(PYTHON) -m ollama_zit_import /path/to/model.safetensors my/imported-model:latest --dry-run
