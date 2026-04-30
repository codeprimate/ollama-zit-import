PYTHON ?= python3
UV ?= uv
VENV_PYTHON ?= .venv/bin/python
UV_PIP_INSTALL ?= $(UV) pip install --python $(VENV_PYTHON)
UV_RUN ?= $(UV) run
BUILD ?= $(PYTHON) -m build
PIP_INSTALL ?= $(PYTHON) -m pip install

.PHONY: install dev-install test lint format typecheck check run-help dry-run-example

install:
	rm -rf dist build
	$(BUILD)
	$(PIP_INSTALL) --force-reinstall dist/*.whl

dev-install:
	$(UV_PIP_INSTALL) -e ".[dev]"

test:
	$(UV_RUN) pytest

lint:
	$(UV_RUN) ruff check .

format:
	$(UV_RUN) black --check .

typecheck:
	$(UV_RUN) mypy src tests

check: lint format typecheck test

run-help:
	$(UV_RUN) python -m ollama_zit_import --help

dry-run-example:
	$(UV_RUN) python -m ollama_zit_import /path/to/model.safetensors my/imported-model:latest --dry-run
