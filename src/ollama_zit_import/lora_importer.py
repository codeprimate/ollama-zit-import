"""LoRA-only derivation execution."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

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
from ollama_zit_import.key_mapping import normalize_base_transformer_name
from ollama_zit_import.lora import LoRASpec
from ollama_zit_import.runtime_support import layer_entry


def execute_lora_derivation(
    *,
    loras: list[LoRASpec],
    base: dict[str, object],
    base_layer_lookup: dict[str, dict[str, object]],
    blobs_dir: str,
    initial_layers: list[dict[str, object]],
    on_transformer_layer_complete: Callable[[], None] | None = None,
) -> tuple[list[dict[str, object]], int, int, list[str]]:
    new_layers = list(initial_layers)
    blobs_new = 0
    blobs_reused = 0
    warnings: list[str] = []

    lora_deltas = load_lora_deltas(loras)
    matched_targets: set[str] = set()
    shape_mismatch_targets: list[str] = []
    raw_layers = cast(list[object], base.get("layers") or [])
    base_layers = [dict(layer) for layer in raw_layers if isinstance(layer, dict)]
    transformer_layers = [
        dict(layer)
        for layer in base_layers
        if str(layer.get("name", "")).startswith("transformer/")
        and str(layer.get("name")) != "transformer/config.json"
    ]
    for layer in transformer_layers:
        layer_name = str(layer.get("name"))
        target_key = normalize_base_transformer_name(layer_name)
        if target_key not in lora_deltas:
            new_layers.append(layer)
            if on_transformer_layer_complete:
                on_transformer_layer_complete()
            continue
        digest = str(layer.get("digest"))
        blob_path = str(Path(blobs_dir) / digest.replace("sha256:", "sha256-"))
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
            scale_blob = str(
                Path(blobs_dir) / str(scale_layer["digest"]).replace("sha256:", "sha256-")
            )
            qbias_blob = str(
                Path(blobs_dir) / str(qbias_layer["digest"]).replace("sha256:", "sha256-")
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
            if on_transformer_layer_complete:
                on_transformer_layer_complete()
            continue
        merged = base_arr + delta
        if base_dtype == "U32":
            if scale_arr is None or qbias_arr is None:
                raise ValueError(f"Missing quantization companions for {layer_name}")
            repacked = quantize_float32_to_u32_packed(merged, scale_arr, qbias_arr, key=layer_name)
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
        if on_transformer_layer_complete:
            on_transformer_layer_complete()
    unmatched_targets = set(lora_deltas.keys()) - matched_targets
    if not matched_targets:
        raise ValueError("No LoRA targets matched base transformer layers")
    if unmatched_targets:
        warnings.append(f"{len(unmatched_targets)} LoRA targets were not matched")
    if shape_mismatch_targets:
        warnings.append(f"{len(shape_mismatch_targets)} LoRA targets had shape mismatches")

    return new_layers, blobs_new, blobs_reused, warnings
