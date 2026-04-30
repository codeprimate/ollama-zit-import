"""Ollama ZIT safetensors importer package."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import tomllib

__all__ = ["__version__"]


def _read_version_from_pyproject() -> str:
    project_root = Path(__file__).resolve().parents[2]
    pyproject_path = project_root / "pyproject.toml"
    with pyproject_path.open("rb") as handle:
        data = tomllib.load(handle)
    return str(data["project"]["version"])


def _resolve_version() -> str:
    package_name = "ollama-zit-import"
    try:
        return version(package_name)
    except PackageNotFoundError:
        return _read_version_from_pyproject()


__version__ = _resolve_version()
