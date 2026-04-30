"""CLI command wiring for ollama-zit-import."""

from __future__ import annotations

import argparse
import os

from rich.console import Console

from ollama_zit_import.importer import run_import


def _common_parser(description: str, *, require_base_model: bool) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--base-model",
        required=require_base_model,
        default=None if require_base_model else "x/z-image-turbo:latest",
        help=(
            "Base model to merge LoRA adapters into"
            if require_base_model
            else "Base model to reuse non-transformer layers from (default: x/z-image-turbo:latest)"
        ),
    )
    parser.add_argument(
        "--ollama-models",
        default=os.environ.get("OLLAMA_MODELS", os.path.expanduser("~/.ollama/models")),
        help="Ollama model store root (default: $OLLAMA_MODELS or ~/.ollama/models)",
    )
    parser.add_argument(
        "--ollama-bin",
        default=None,
        help="Optional explicit path to ollama binary (auto-detected if omitted)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and print plan, but do not write blobs/manifest",
    )
    return parser


def parse_args() -> argparse.Namespace:
    description = "Import z-image-turbo safetensors into Ollama as a new model."
    argv = os.sys.argv[1:]
    if "--lora" in argv:
        parser = _common_parser(description, require_base_model=True)
        parser.add_argument(
            "output_model",
            help="Output Ollama model name (examples: custom-model, me/custom-model:latest)",
        )
        parser.add_argument(
            "--lora",
            action="append",
            required=True,
            help="LoRA adapter input in PATH@WEIGHT form (repeatable)",
        )
        return parser.parse_args(argv)

    parser = _common_parser(description, require_base_model=False)
    parser.add_argument("safetensors_path", help="Path to source .safetensors checkpoint")
    parser.add_argument(
        "output_model",
        help="Output Ollama model name (examples: custom-model, me/custom-model:latest)",
    )
    return parser.parse_args(argv)


def run() -> int:
    from ollama_zit_import import __version__

    console = Console()
    console.print(f"[bold]ollama-zit-import[/bold] [dim]v{__version__}[/dim]")
    console.rule(style="dim")
    console.print()

    return run_import(parse_args(), console=console)
