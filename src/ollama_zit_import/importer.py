"""Reusable importer orchestration layer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from ollama_zit_import.derive import read_safetensors_header
from ollama_zit_import.key_mapping import count_matching_targets
from ollama_zit_import.lora import analyze_lora_header
from ollama_zit_import.lora_importer import execute_lora_derivation
from ollama_zit_import.model_importer import execute_standard_import
from ollama_zit_import.planning import create_import_plan
from ollama_zit_import.runtime_support import (
    check_ollama_binary,
    detect_ollama_binary,
    ensure_base_model_present,
)

CONFIG_LAYER_NAME = "transformer/config.json"
TRANSFORMER_PREFIX = "transformer/"


def _validate_inputs(plan, ollama_models: Path) -> None:
    if plan.mode == "standard_import":
        if plan.safetensors_path is None:
            raise ValueError("Missing safetensors path in standard import mode")
        if not Path(plan.safetensors_path).is_file():
            raise FileNotFoundError(f"Safetensors file not found: {plan.safetensors_path}")
        if not plan.safetensors_path.endswith(".safetensors"):
            raise ValueError("Input must be a .safetensors file")
    else:
        for lora in plan.loras:
            if not Path(lora.path).is_file():
                raise FileNotFoundError(f"LoRA file not found: {lora.path}")

    if not ollama_models.is_dir():
        raise FileNotFoundError(f"Ollama models directory not found: {ollama_models}")


def _kept_base_layers(base: dict[str, object]) -> list[dict[str, object]]:
    kept_layers: list[dict[str, object]] = []
    for layer in base.get("layers", []):
        name = str(layer.get("name", ""))
        component = name.split("/")[0] if "/" in name else ""
        if component != "transformer":
            kept_layers.append(dict(layer))

    config_kept = False
    for layer in base.get("layers", []):
        if layer.get("name") == CONFIG_LAYER_NAME:
            kept_layers.append(dict(layer))
            config_kept = True
            break
    if not config_kept:
        raise RuntimeError("Base manifest did not contain transformer/config.json")
    return kept_layers


def run_import(args: argparse.Namespace, *, console: Console | None = None) -> int:
    active_console = console
    plan = create_import_plan(args)
    ollama_models = Path(args.ollama_models).resolve()
    blobs_dir = ollama_models / "blobs"
    manifests_root = ollama_models / "manifests" / "registry.ollama.ai"

    _validate_inputs(plan, ollama_models)

    ollama_bin = detect_ollama_binary(args.ollama_bin)
    check_ollama_binary(ollama_bin)

    base_model = plan.base_model
    output_model = plan.output_model

    base_manifest = ensure_base_model_present(
        ollama_bin=ollama_bin,
        base_model=base_model,
        manifests_root=str(manifests_root),
        blobs_dir=str(blobs_dir),
        console=active_console,
    )
    out_manifest = manifests_root / output_model.manifest_path_part
    out_manifest_dir = out_manifest.parent

    if not blobs_dir.is_dir():
        raise FileNotFoundError(f"Ollama blobs directory missing: {blobs_dir}")

    with Path(base_manifest).open(encoding="utf-8") as handle:
        base = json.load(handle)
    base_layer_lookup = {
        str(layer.get("name")): dict(layer) for layer in base.get("layers", []) if layer.get("name")
    }
    kept_layers = _kept_base_layers(base)

    source_tensors: list[str] = []
    header: dict[str, object] = {}
    data_offset = 0
    safetensors_path: str | None = None
    matched_lora_keys = 0
    unmatched_lora_keys = 0
    estimated_tensor_writes = 0
    base_transformer_names = [
        str(layer.get("name"))
        for layer in base.get("layers", [])
        if str(layer.get("name", "")).startswith(TRANSFORMER_PREFIX)
        and str(layer.get("name")) != CONFIG_LAYER_NAME
    ]
    if plan.mode == "standard_import":
        if plan.safetensors_path is None:
            raise ValueError("Missing safetensors path in standard import mode")
        safetensors_path = plan.safetensors_path
        header, data_offset = read_safetensors_header(safetensors_path)
        source_tensors = sorted(key for key in header if key != "__metadata__")
        estimated_tensor_writes = len(source_tensors)
        if active_console:
            active_console.print(f"  [bold]Checkpoint[/bold]    {safetensors_path}")
            active_console.print(f"                [dim]{len(source_tensors)} transformer tensors[/dim]")
            active_console.print()
    else:
        if active_console:
            active_console.print("  [bold]Mode[/bold]          [cyan]LoRA merge import[/cyan]")
            active_console.print(f"                [dim]{len(plan.loras)} LoRA adapter(s)[/dim]")
        for lora_spec in plan.loras:
            lora_header, _ = read_safetensors_header(lora_spec.path)
            lora_analysis = analyze_lora_header(lora_header)
            matched = count_matching_targets(
                base_layer_names=base_transformer_names,
                lora_target_keys=lora_analysis.recognized_target_keys,
            )
            matched_lora_keys += matched
            unmatched_lora_keys += lora_analysis.unmatched_count + (lora_analysis.matched_count - matched)
            if active_console:
                active_console.print(
                    f"                [dim]- {lora_spec.path} @ {lora_spec.user_weight:.3f}[/dim]"
                )
                active_console.print(
                    "                "
                    f"[dim]  targets: {lora_analysis.matched_count} parsed, "
                    f"{matched} base-matched, {lora_analysis.unmatched_count} unparsed[/dim]"
                )
        estimated_tensor_writes = matched_lora_keys
        if active_console:
            active_console.print()
    if active_console:
        active_console.print(f"  [bold]Base model[/bold]    [cyan]{base_model.display_name}[/cyan]")
        active_console.print(f"                [dim]{len(kept_layers)} non-transformer layers retained[/dim]")
        active_console.print()
        active_console.print(f"  [bold]Output[/bold]        [cyan]{output_model.display_name}[/cyan]")
        active_console.print(f"                [dim]{out_manifest}[/dim]")
        active_console.print()
        active_console.rule(style="dim")

    if args.dry_run:
        if active_console:
            active_console.print()
            active_console.print("  [bold yellow]Dry run[/bold yellow] — no files written. Looks good.")
            active_console.print(f"                [dim]branch={plan.mode}[/dim]")
            active_console.print(
                "                "
                f"[dim]matched_keys={matched_lora_keys} unmatched_keys={unmatched_lora_keys} "
                f"estimated_writes={estimated_tensor_writes}[/dim]"
            )
            active_console.print()
            active_console.rule(style="dim")
        return 0

    if out_manifest.exists():
        raise FileExistsError(f"Output manifest already exists: {out_manifest}")

    out_manifest_dir.mkdir(parents=True, exist_ok=True)
    blobs_new = 0
    blobs_reused = 0

    if plan.mode == "lora_only_derivation":
        if active_console:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(bar_width=None),
                MofNCompleteColumn(),
                TextColumn("{task.percentage:>3.0f}%"),
                TimeElapsedColumn(),
                TimeRemainingColumn(),
                console=active_console,
            ) as progress:
                task_id = progress.add_task("Applying LoRA deltas", total=len(base_transformer_names))
                new_layers, blobs_new, blobs_reused, warnings = execute_lora_derivation(
                    loras=plan.loras,
                    base=base,
                    base_layer_lookup=base_layer_lookup,
                    blobs_dir=str(blobs_dir),
                    initial_layers=kept_layers,
                    on_transformer_layer_complete=lambda: progress.advance(task_id),
                )
        else:
            new_layers, blobs_new, blobs_reused, warnings = execute_lora_derivation(
                loras=plan.loras,
                base=base,
                base_layer_lookup=base_layer_lookup,
                blobs_dir=str(blobs_dir),
                initial_layers=kept_layers,
            )
        if active_console:
            for warning in warnings:
                active_console.print(f"[yellow]Warning:[/yellow] {warning}")
    else:
        if safetensors_path is None:
            raise ValueError("Missing safetensors path in standard import mode")
        if active_console:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(bar_width=None),
                MofNCompleteColumn(),
                TextColumn("{task.percentage:>3.0f}%"),
                TimeElapsedColumn(),
                TimeRemainingColumn(),
                console=active_console,
            ) as progress:
                task_id = progress.add_task("Converting and writing tensors", total=len(source_tensors))
                new_layers, blobs_new, blobs_reused = execute_standard_import(
                    source_tensors=source_tensors,
                    header=header,
                    data_offset=data_offset,
                    safetensors_path=safetensors_path,
                    blobs_dir=str(blobs_dir),
                    initial_layers=kept_layers,
                    on_source_tensor_complete=lambda: progress.advance(task_id),
                )
        else:
            new_layers, blobs_new, blobs_reused = execute_standard_import(
                source_tensors=source_tensors,
                header=header,
                data_offset=data_offset,
                safetensors_path=safetensors_path,
                blobs_dir=str(blobs_dir),
                initial_layers=kept_layers,
            )

    new_manifest = {
        "schemaVersion": base["schemaVersion"],
        "mediaType": base["mediaType"],
        "config": base["config"],
        "layers": new_layers,
    }
    with out_manifest.open("w", encoding="utf-8") as handle:
        json.dump(new_manifest, handle, indent=2)

    total_layers = len(new_layers)
    base_count = len(kept_layers)
    transformer_count = total_layers - base_count

    if active_console:
        active_console.print()
        active_console.print(f"  [bold green]Imported[/bold green]  [cyan]{output_model.display_name}[/cyan]")
        active_console.print()
        active_console.print(
            f"  [bold]Layers[/bold]    {total_layers} total"
            f"  [dim]({base_count} base + {transformer_count} transformer)[/dim]"
        )
        active_console.print(f"  [bold]Blobs[/bold]     {blobs_new} written · {blobs_reused} reused")
        active_console.print()
        active_console.print(
            f'  [bold]Run:[/bold]  [cyan]{ollama_bin} run {output_model.run_name} "your prompt"[/cyan]'
        )
        active_console.print()
        active_console.rule(style="dim")
    return 0
