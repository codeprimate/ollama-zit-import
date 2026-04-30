"""Shared runtime support helpers for import execution."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

from rich.console import Console

from ollama_zit_import.planning import ModelRef

OLLAMA_BINARY_CANDIDATES = (
    "/usr/local/bin/ollama",
    "/opt/homebrew/bin/ollama",
    "/Applications/Ollama.app/Contents/Resources/ollama",
)
OLLAMA_VERSION_FLAG = "--version"
OLLAMA_PULL_COMMAND = "pull"
SHA256_PREFIX = "sha256:"
SHA256_BLOB_PREFIX = "sha256-"
LAYER_MEDIA_TYPE = "application/vnd.ollama.image.tensor"


def detect_ollama_binary(explicit: str | None) -> str:
    if explicit:
        explicit_path = Path(explicit)
        if explicit_path.is_file() and os.access(explicit_path, os.X_OK):
            return explicit
        raise FileNotFoundError(f"Provided --ollama-bin is not executable: {explicit}")

    from_path = shutil.which("ollama")
    if from_path:
        return from_path

    for candidate in OLLAMA_BINARY_CANDIDATES:
        candidate_path = Path(candidate)
        if candidate_path.is_file() and os.access(candidate_path, os.X_OK):
            return candidate

    raise FileNotFoundError("Could not locate ollama binary. Add it to PATH or pass --ollama-bin.")


def check_ollama_binary(ollama_bin: str) -> None:
    result = subprocess.run(
        [ollama_bin, OLLAMA_VERSION_FLAG],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"ollama binary failed --version check ({result.returncode}): {result.stderr.strip()}"
        )


def _blob_path(blobs_dir: str, digest: str) -> str:
    return str(Path(blobs_dir) / digest.replace(SHA256_PREFIX, SHA256_BLOB_PREFIX))


def ensure_base_model_present(
    ollama_bin: str,
    base_model: ModelRef,
    manifests_root: str,
    blobs_dir: str,
    console: Console | None,
) -> str:
    base_manifest_path = Path(manifests_root) / base_model.manifest_path_part
    needs_pull = not base_manifest_path.is_file()

    if not needs_pull:
        with base_manifest_path.open(encoding="utf-8") as handle:
            manifest = json.load(handle)
        for layer in manifest.get("layers", []):
            digest = layer.get("digest")
            if not digest:
                continue
            if not Path(_blob_path(blobs_dir, str(digest))).is_file():
                needs_pull = True
                break

    if needs_pull:
        if console:
            console.print(
                f"[bold yellow]Base model missing/incomplete.[/bold yellow] "
                f"Pulling [cyan]{base_model.display_name}[/cyan] ..."
            )
        result = subprocess.run(
            [ollama_bin, OLLAMA_PULL_COMMAND, base_model.display_name], check=False
        )
        if result.returncode != 0:
            raise RuntimeError(f"Failed to pull base model: {base_model.display_name}")
        if not base_manifest_path.is_file():
            raise FileNotFoundError(f"Base manifest still missing after pull: {base_manifest_path}")

    return str(base_manifest_path)


def layer_entry(name: str, digest: str, size: int) -> dict[str, object]:
    return {
        "mediaType": LAYER_MEDIA_TYPE,
        "digest": digest,
        "size": size,
        "name": name,
    }
