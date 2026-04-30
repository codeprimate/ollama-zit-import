"""Execution planning and model reference helpers."""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Literal

from ollama_zit_import.lora import LoRASpec, parse_lora_specs

DEFAULT_MODEL_NAMESPACE = "my"
ExecutionMode = Literal["standard_import", "lora_only_derivation"]


@dataclass(frozen=True)
class ModelRef:
    namespace: str
    name: str
    tag: str

    @property
    def manifest_path_part(self) -> str:
        return os.path.join(self.namespace, self.name, self.tag)

    @property
    def display_name(self) -> str:
        core = f"{self.namespace}/{self.name}"
        return core if self.tag == "latest" else f"{core}:{self.tag}"

    @property
    def run_name(self) -> str:
        core = f"{self.namespace}/{self.name}"
        return core if self.tag == "latest" else f"{core}:{self.tag}"


@dataclass(frozen=True)
class ImportPlan:
    mode: ExecutionMode
    output_model: ModelRef
    base_model: ModelRef
    safetensors_path: str | None
    loras: list[LoRASpec]


def parse_model_ref(model: str) -> ModelRef:
    model = model.strip()
    if not model:
        raise ValueError("Model name cannot be empty")

    if ":" in model:
        core, tag = model.rsplit(":", 1)
        tag = tag or "latest"
    else:
        core, tag = model, "latest"

    if "/" in core:
        namespace, name = core.split("/", 1)
    else:
        namespace, name = DEFAULT_MODEL_NAMESPACE, core

    if not namespace or not name:
        raise ValueError(f"Invalid model reference: {model}")
    return ModelRef(namespace=namespace, name=name, tag=tag)


def determine_mode(args: argparse.Namespace) -> ExecutionMode:
    return "lora_only_derivation" if getattr(args, "lora", None) else "standard_import"


def create_import_plan(args: argparse.Namespace) -> ImportPlan:
    mode = determine_mode(args)
    output_model = parse_model_ref(args.output_model)
    base_model = parse_model_ref(args.base_model)

    if mode == "standard_import":
        safetensors_path = os.path.abspath(args.safetensors_path)
        return ImportPlan(
            mode=mode,
            output_model=output_model,
            base_model=base_model,
            safetensors_path=safetensors_path,
            loras=[],
        )

    loras = parse_lora_specs(args.lora)
    return ImportPlan(
        mode=mode,
        output_model=output_model,
        base_model=base_model,
        safetensors_path=None,
        loras=loras,
    )

