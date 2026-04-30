"""Import a z-image-turbo-compatible safetensors checkpoint into Ollama."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from typing import Any, cast

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

from ollama_zit_import.derive import (
    load_lora_deltas,
    make_single_tensor_blob,
    quantize_float32_to_u32_packed,
    read_safetensors_header,
    read_tensor_raw,
    to_bf16_bytes,
    to_float32,
    write_blob,
)
from ollama_zit_import.key_mapping import (
    count_matching_targets,
    map_name,
    normalize_base_transformer_name,
)
from ollama_zit_import.lora import analyze_lora_header
from ollama_zit_import.planning import ModelRef, create_import_plan


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


def detect_ollama_binary(explicit: str | None) -> str:
    if explicit:
        if os.path.isfile(explicit) and os.access(explicit, os.X_OK):
            return explicit
        raise FileNotFoundError(f"Provided --ollama-bin is not executable: {explicit}")

    from_path = shutil.which("ollama")
    if from_path:
        return from_path

    candidates = [
        "/usr/local/bin/ollama",
        "/opt/homebrew/bin/ollama",
        "/Applications/Ollama.app/Contents/Resources/ollama",
    ]
    for candidate in candidates:
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate

    raise FileNotFoundError("Could not locate ollama binary. Add it to PATH or pass --ollama-bin.")


def check_ollama_binary(ollama_bin: str) -> None:
    result = subprocess.run(
        [ollama_bin, "--version"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"ollama binary failed --version check ({result.returncode}): {result.stderr.strip()}"
        )


def ensure_base_model_present(
    ollama_bin: str,
    base_model: ModelRef,
    manifests_root: str,
    blobs_dir: str,
    console: Console,
) -> str:
    base_manifest = os.path.join(manifests_root, base_model.manifest_path_part)
    needs_pull = not os.path.isfile(base_manifest)

    if not needs_pull:
        with open(base_manifest, "r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        for layer in manifest.get("layers", []):
            digest = layer.get("digest")
            if not digest:
                continue
            blob_path = os.path.join(blobs_dir, digest.replace("sha256:", "sha256-"))
            if not os.path.isfile(blob_path):
                needs_pull = True
                break

    if needs_pull:
        console.print(
            f"[bold yellow]Base model missing/incomplete.[/bold yellow] "
            f"Pulling [cyan]{base_model.display_name}[/cyan] ..."
        )
        result = subprocess.run([ollama_bin, "pull", base_model.display_name], check=False)
        if result.returncode != 0:
            raise RuntimeError(f"Failed to pull base model: {base_model.display_name}")
        if not os.path.isfile(base_manifest):
            raise FileNotFoundError(f"Base manifest still missing after pull: {base_manifest}")

    return base_manifest


def layer_entry(name: str, digest: str, size: int) -> dict[str, object]:
    return {
        "mediaType": "application/vnd.ollama.image.tensor",
        "digest": digest,
        "size": size,
        "name": name,
    }


def run() -> int:
    from ollama_zit_import import __version__

    console = Console()
    console.print(f"[bold]ollama-zit-import[/bold] [dim]v{__version__}[/dim]")
    console.rule(style="dim")
    console.print()

    args = parse_args()
    plan = create_import_plan(args)
    ollama_models = os.path.abspath(args.ollama_models)
    blobs_dir = os.path.join(ollama_models, "blobs")
    manifests_root = os.path.join(ollama_models, "manifests", "registry.ollama.ai")

    if plan.mode == "standard_import":
        if plan.safetensors_path is None:
            raise ValueError("Missing safetensors path in standard import mode")
        if not os.path.isfile(plan.safetensors_path):
            raise FileNotFoundError(f"Safetensors file not found: {plan.safetensors_path}")
        if not plan.safetensors_path.endswith(".safetensors"):
            raise ValueError("Input must be a .safetensors file")
    else:
        for lora in plan.loras:
            if not os.path.isfile(lora.path):
                raise FileNotFoundError(f"LoRA file not found: {lora.path}")

    if not os.path.isdir(ollama_models):
        raise FileNotFoundError(f"Ollama models directory not found: {ollama_models}")

    ollama_bin = detect_ollama_binary(args.ollama_bin)
    check_ollama_binary(ollama_bin)

    base_model = plan.base_model
    output_model = plan.output_model

    base_manifest = ensure_base_model_present(
        ollama_bin=ollama_bin,
        base_model=base_model,
        manifests_root=manifests_root,
        blobs_dir=blobs_dir,
        console=console,
    )
    out_manifest = os.path.join(manifests_root, output_model.manifest_path_part)
    out_manifest_dir = os.path.dirname(out_manifest)

    if not os.path.isdir(blobs_dir):
        raise FileNotFoundError(f"Ollama blobs directory missing: {blobs_dir}")

    with open(base_manifest, "r", encoding="utf-8") as handle:
        base = json.load(handle)
    base_layer_lookup = {
        str(layer.get("name")): dict(layer) for layer in base.get("layers", []) if layer.get("name")
    }

    kept_layers: list[dict[str, object]] = []
    for layer in base.get("layers", []):
        name = str(layer.get("name", ""))
        component = name.split("/")[0] if "/" in name else ""
        if component != "transformer":
            kept_layers.append(dict(layer))

    config_kept = False
    for layer in base.get("layers", []):
        if layer.get("name") == "transformer/config.json":
            kept_layers.append(dict(layer))
            config_kept = True
            break
    if not config_kept:
        raise RuntimeError("Base manifest did not contain transformer/config.json")

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
        if str(layer.get("name", "")).startswith("transformer/")
        and str(layer.get("name")) != "transformer/config.json"
    ]
    if plan.mode == "standard_import":
        if plan.safetensors_path is None:
            raise ValueError("Missing safetensors path in standard import mode")
        safetensors_path = plan.safetensors_path
        header, data_offset = read_safetensors_header(safetensors_path)
        source_tensors = sorted(key for key in header.keys() if key != "__metadata__")
        estimated_tensor_writes = len(source_tensors)
        console.print(f"  [bold]Checkpoint[/bold]    {safetensors_path}")
        console.print(f"                [dim]{len(source_tensors)} transformer tensors[/dim]")
        console.print()
    else:
        console.print("  [bold]Mode[/bold]          [cyan]LoRA merge import[/cyan]")
        console.print(f"                [dim]{len(plan.loras)} LoRA adapter(s)[/dim]")
        for lora_spec in plan.loras:
            lora_header, _ = read_safetensors_header(lora_spec.path)
            lora_analysis = analyze_lora_header(lora_header)
            matched = count_matching_targets(
                base_layer_names=base_transformer_names,
                lora_target_keys=lora_analysis.recognized_target_keys,
            )
            matched_lora_keys += matched
            unmatched_lora_keys += lora_analysis.unmatched_count + (
                lora_analysis.matched_count - matched
            )
            console.print(
                f"                [dim]- {lora_spec.path} @ {lora_spec.user_weight:.3f}[/dim]"
            )
            console.print(
                "                "
                f"[dim]  targets: {lora_analysis.matched_count} parsed, "
                f"{matched} base-matched, {lora_analysis.unmatched_count} unparsed[/dim]"
            )
        estimated_tensor_writes = matched_lora_keys
        console.print()
    console.print(f"  [bold]Base model[/bold]    [cyan]{base_model.display_name}[/cyan]")
    console.print(f"                [dim]{len(kept_layers)} non-transformer layers retained[/dim]")
    console.print()
    console.print(f"  [bold]Output[/bold]        [cyan]{output_model.display_name}[/cyan]")
    console.print(f"                [dim]{out_manifest}[/dim]")
    console.print()
    console.rule(style="dim")

    if args.dry_run:
        console.print()
        console.print("  [bold yellow]Dry run[/bold yellow] — no files written. Looks good.")
        console.print(f"                [dim]branch={plan.mode}[/dim]")
        console.print(
            "                "
            f"[dim]matched_keys={matched_lora_keys} unmatched_keys={unmatched_lora_keys} "
            f"estimated_writes={estimated_tensor_writes}[/dim]"
        )
        console.print()
        console.rule(style="dim")
        return 0

    if os.path.exists(out_manifest):
        raise FileExistsError(f"Output manifest already exists: {out_manifest}")

    os.makedirs(out_manifest_dir, exist_ok=True)
    new_layers = list(kept_layers)
    blobs_new = 0
    blobs_reused = 0

    if plan.mode == "lora_only_derivation":
        lora_deltas = load_lora_deltas(plan.loras)
        matched_targets: set[str] = set()
        shape_mismatch_targets: list[str] = []
        transformer_layers = [
            dict(layer)
            for layer in base.get("layers", [])
            if str(layer.get("name", "")).startswith("transformer/")
            and str(layer.get("name")) != "transformer/config.json"
        ]
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(bar_width=None),
            MofNCompleteColumn(),
            TextColumn("{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=console,
        ) as progress:
            task_id = progress.add_task("Applying LoRA deltas", total=len(transformer_layers))
            for layer in transformer_layers:
                layer_name = str(layer.get("name"))
                target_key = normalize_base_transformer_name(layer_name)
                if target_key not in lora_deltas:
                    new_layers.append(layer)
                    progress.advance(task_id)
                    continue
                digest = str(layer.get("digest"))
                blob_path = os.path.join(blobs_dir, digest.replace("sha256:", "sha256-"))
                layer_header, layer_offset = read_safetensors_header(blob_path)
                layer_info = cast(dict[str, Any], layer_header["data"])
                base_shape = cast(list[int], layer_info["shape"])
                base_dtype = cast(str, layer_info["dtype"])
                base_raw = read_tensor_raw(blob_path, layer_header, "data", layer_offset)
                scale_arr = None
                qbias_arr = None
                if base_dtype == "U32":
                    scale_layer_name = layer_name.removesuffix(".weight") + ".weight_scale"
                    qbias_layer_name = layer_name.removesuffix(".weight") + ".weight_qbias"
                    scale_layer = base_layer_lookup.get(scale_layer_name)
                    qbias_layer = base_layer_lookup.get(qbias_layer_name)
                    if scale_layer is None or qbias_layer is None:
                        raise ValueError(
                            f"Missing quantization companions for {layer_name}: "
                            f"{scale_layer_name}, {qbias_layer_name}"
                        )
                    scale_blob = os.path.join(
                        blobs_dir, str(scale_layer["digest"]).replace("sha256:", "sha256-")
                    )
                    qbias_blob = os.path.join(
                        blobs_dir, str(qbias_layer["digest"]).replace("sha256:", "sha256-")
                    )
                    s_header, s_offset = read_safetensors_header(scale_blob)
                    q_header, q_offset = read_safetensors_header(qbias_blob)
                    s_info = cast(dict[str, Any], s_header["data"])
                    q_info = cast(dict[str, Any], q_header["data"])
                    s_raw = read_tensor_raw(scale_blob, s_header, "data", s_offset)
                    q_raw = read_tensor_raw(qbias_blob, q_header, "data", q_offset)
                    scale_arr = to_float32(
                        s_raw,
                        cast(str, s_info["dtype"]),
                        cast(list[int], s_info["shape"]),
                        key=scale_layer_name,
                    )
                    qbias_arr = to_float32(
                        q_raw,
                        cast(str, q_info["dtype"]),
                        cast(list[int], q_info["shape"]),
                        key=qbias_layer_name,
                    )
                base_arr = to_float32(
                    base_raw,
                    base_dtype,
                    base_shape,
                    scale=scale_arr,
                    qbias=qbias_arr,
                    key=layer_name,
                )
                delta = lora_deltas[target_key]
                if list(delta.shape) != list(base_arr.shape):
                    shape_mismatch_targets.append(
                        f"{target_key} delta={list(delta.shape)} base={list(base_arr.shape)}"
                    )
                    new_layers.append(layer)
                    progress.advance(task_id)
                    continue
                merged = base_arr + delta
                if base_dtype == "U32":
                    if scale_arr is None or qbias_arr is None:
                        raise ValueError(f"Missing quantization companions for {layer_name}")
                    repacked = quantize_float32_to_u32_packed(
                        merged, scale_arr, qbias_arr, key=layer_name
                    )
                    out_blob = make_single_tensor_blob("U32", base_shape, repacked.tobytes())
                else:
                    out_blob = make_single_tensor_blob("BF16", list(merged.shape), to_bf16_bytes(merged))
                out_digest, out_size, is_new = write_blob(blobs_dir, out_blob)
                new_layers.append(layer_entry(layer_name, out_digest, out_size))
                if is_new:
                    blobs_new += 1
                else:
                    blobs_reused += 1
                matched_targets.add(target_key)
                progress.advance(task_id)
        unmatched_targets = set(lora_deltas.keys()) - matched_targets
        if not matched_targets:
            raise ValueError("No LoRA targets matched base transformer layers")
        if unmatched_targets:
            console.print(
                f"[yellow]Warning:[/yellow] {len(unmatched_targets)} LoRA targets were not matched"
            )
        if shape_mismatch_targets:
            console.print(
                f"[yellow]Warning:[/yellow] {len(shape_mismatch_targets)} LoRA targets had shape mismatches"
            )
    else:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(bar_width=None),
            MofNCompleteColumn(),
            TextColumn("{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=console,
        ) as progress:
            task_id = progress.add_task("Converting and writing tensors", total=len(source_tensors))
            for source_name in source_tensors:
                info = cast(dict[str, Any], header[source_name])
                dtype = cast(str, info["dtype"])
                shape = cast(list[int], info["shape"])
                raw = read_tensor_raw(safetensors_path, header, source_name, data_offset)

                mapped = map_name(source_name)
                for target_name, qkv_part in mapped:
                    if qkv_part is None:
                        if dtype == "BF16":
                            bf16_data = raw
                            out_shape = list(shape)
                        else:
                            arr = to_float32(raw, dtype, shape)
                            bf16_data = to_bf16_bytes(arr)
                            out_shape = list(arr.shape)
                    else:
                        arr = to_float32(raw, dtype, shape)
                        if len(shape) < 1 or shape[0] % 3 != 0:
                            raise ValueError(
                                f"QKV tensor has unsupported shape for split ({shape}): {source_name}"
                            )
                        chunk = shape[0] // 3
                        splits = {
                            "q": arr[:chunk],
                            "k": arr[chunk : 2 * chunk],
                            "v": arr[2 * chunk :],
                        }
                        out_arr = splits[qkv_part]
                        bf16_data = to_bf16_bytes(out_arr)
                        out_shape = list(out_arr.shape)

                    digest, size, is_new = write_blob(
                        blobs_dir,
                        make_single_tensor_blob("BF16", out_shape, bf16_data),
                    )
                    new_layers.append(layer_entry(f"transformer/{target_name}", digest, size))
                    if is_new:
                        blobs_new += 1
                    else:
                        blobs_reused += 1
                progress.advance(task_id)

    new_manifest = {
        "schemaVersion": base["schemaVersion"],
        "mediaType": base["mediaType"],
        "config": base["config"],
        "layers": new_layers,
    }
    with open(out_manifest, "w", encoding="utf-8") as handle:
        json.dump(new_manifest, handle, indent=2)

    total_layers = len(new_layers)
    base_count = len(kept_layers)
    transformer_count = total_layers - base_count

    console.print()
    console.print(f"  [bold green]Imported[/bold green]  [cyan]{output_model.display_name}[/cyan]")
    console.print()
    console.print(
        f"  [bold]Layers[/bold]    {total_layers} total"
        f"  [dim]({base_count} base + {transformer_count} transformer)[/dim]"
    )
    console.print(f"  [bold]Blobs[/bold]     {blobs_new} written · {blobs_reused} reused")
    console.print()
    console.print(
        f'  [bold]Run:[/bold]  [cyan]{ollama_bin} run {output_model.run_name} "your prompt"[/cyan]'
    )
    console.print()
    console.rule(style="dim")
    return 0
