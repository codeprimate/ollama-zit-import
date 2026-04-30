"""Standard checkpoint import execution."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast

from ollama_zit_import.derive import (
    make_single_tensor_blob,
    read_tensor_raw,
    to_bf16_bytes,
    to_float32,
    write_blob,
)
from ollama_zit_import.key_mapping import map_name
from ollama_zit_import.runtime_support import layer_entry


def execute_standard_import(
    *,
    source_tensors: list[str],
    header: dict[str, object],
    data_offset: int,
    safetensors_path: str,
    blobs_dir: str,
    initial_layers: list[dict[str, object]],
    on_source_tensor_complete: Callable[[], None] | None = None,
) -> tuple[list[dict[str, object]], int, int]:
    new_layers = list(initial_layers)
    blobs_new = 0
    blobs_reused = 0

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
        if on_source_tensor_complete:
            on_source_tensor_complete()

    return new_layers, blobs_new, blobs_reused
