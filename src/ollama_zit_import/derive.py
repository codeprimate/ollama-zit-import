"""Tensor decode/encode and LoRA merge helpers."""

from __future__ import annotations

import hashlib
import json
import struct
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np

from ollama_zit_import.lora import LoRASpec, parse_lora_tensor_key


@dataclass(frozen=True)
class DeriveStats:
    matched_keys: int
    unmatched_keys: int
    written_blobs: int
    reused_blobs: int


def read_safetensors_header(path: str) -> tuple[dict[str, object], int]:
    with Path(path).open("rb") as handle:
        header_len = struct.unpack("<Q", handle.read(8))[0]
        header = json.loads(handle.read(header_len))
    return cast(dict[str, object], header), 8 + header_len


def read_tensor_raw(
    path: str, header: dict[str, object], tensor_name: str, data_offset: int
) -> bytes:
    tensor_entry = cast(dict[str, Any], header[tensor_name])
    start, end = cast(tuple[int, int], tuple(tensor_entry["data_offsets"]))
    with Path(path).open("rb") as handle:
        handle.seek(data_offset + start)
        return handle.read(end - start)


def _dequantize_u32_with_scale_bias(
    packed: np.ndarray, scale: np.ndarray, qbias: np.ndarray, key: str
) -> np.ndarray:
    if packed.ndim != 2 or scale.ndim != 2 or qbias.ndim != 2:
        raise ValueError(f"U32 dequantization expects 2D tensors for {key}")
    if scale.shape != qbias.shape:
        raise ValueError(
            f"U32 scale/qbias shape mismatch for {key}: {scale.shape} vs {qbias.shape}"
        )
    if packed.shape[0] != scale.shape[0]:
        raise ValueError(f"U32 row mismatch for {key}: packed={packed.shape}, scale={scale.shape}")
    unpacked = np.empty((packed.shape[0], packed.shape[1] * 4), dtype=np.uint8)
    unpacked[:, 0::4] = (packed & 0xFF).astype(np.uint8)
    unpacked[:, 1::4] = ((packed >> 8) & 0xFF).astype(np.uint8)
    unpacked[:, 2::4] = ((packed >> 16) & 0xFF).astype(np.uint8)
    unpacked[:, 3::4] = ((packed >> 24) & 0xFF).astype(np.uint8)

    if unpacked.shape[1] % scale.shape[1] != 0:
        raise ValueError(
            f"U32 columns are not divisible by scale groups for {key}: "
            f"packed={packed.shape}, unpacked={unpacked.shape}, scale={scale.shape}"
        )
    group_width = unpacked.shape[1] // scale.shape[1]
    reshaped = unpacked.astype(np.float32).reshape(unpacked.shape[0], scale.shape[1], group_width)
    out = (reshaped - qbias[..., None]) * scale[..., None]
    return cast(np.ndarray, out.reshape(unpacked.shape).astype(np.float32))


def quantize_float32_to_u32_packed(
    values: np.ndarray, scale: np.ndarray, qbias: np.ndarray, key: str
) -> np.ndarray:
    if values.ndim != 2 or scale.ndim != 2 or qbias.ndim != 2:
        raise ValueError(f"U32 quantization expects 2D tensors for {key}")
    if scale.shape != qbias.shape:
        raise ValueError(
            f"U32 scale/qbias shape mismatch for {key}: {scale.shape} vs {qbias.shape}"
        )
    if values.shape[0] != scale.shape[0]:
        raise ValueError(f"U32 row mismatch for {key}: values={values.shape}, scale={scale.shape}")
    if values.shape[1] % scale.shape[1] != 0:
        raise ValueError(
            f"U32 columns are not divisible by scale groups for {key}: "
            f"values={values.shape}, scale={scale.shape}"
        )
    group_width = values.shape[1] // scale.shape[1]
    if group_width % 4 != 0:
        raise ValueError(
            f"U32 group width must be divisible by 4 for {key}: "
            f"values={values.shape}, scale={scale.shape}"
        )
    grouped = values.reshape(values.shape[0], scale.shape[1], group_width)
    quantized = np.rint((grouped / scale[..., None]) + qbias[..., None])
    quantized = np.clip(quantized, 0, 255).astype(np.uint8).reshape(values.shape)
    packed = (
        quantized[:, 0::4].astype(np.uint32)
        | (quantized[:, 1::4].astype(np.uint32) << 8)
        | (quantized[:, 2::4].astype(np.uint32) << 16)
        | (quantized[:, 3::4].astype(np.uint32) << 24)
    )
    return cast(np.ndarray, packed)


def to_float32(
    raw: bytes,
    dtype: str,
    shape: list[int],
    *,
    scale: np.ndarray | None = None,
    qbias: np.ndarray | None = None,
    key: str = "",
) -> np.ndarray:
    if dtype == "BF16":
        u16 = np.frombuffer(raw, dtype=np.uint16)
        return (u16.astype(np.uint32) << 16).view(np.float32).reshape(shape)
    if dtype == "F16":
        return np.frombuffer(raw, dtype=np.float16).astype(np.float32).reshape(shape)
    if dtype == "F32":
        return np.frombuffer(raw, dtype=np.float32).reshape(shape)
    if dtype == "U32":
        packed = np.frombuffer(raw, dtype=np.uint32).reshape(shape)
        if scale is None or qbias is None:
            raise ValueError(
                f"Unsupported source dtype for tensor conversion without scale/qbias: {dtype}"
            )
        return _dequantize_u32_with_scale_bias(packed, scale, qbias, key)
    raise ValueError(f"Unsupported source dtype for tensor conversion: {dtype}")


def to_bf16_bytes(arr: np.ndarray) -> bytes:
    return (arr.astype(np.float32).view(np.uint32) >> 16).astype(np.uint16).tobytes()


def make_single_tensor_blob(dtype: str, shape: list[int], tensor_data: bytes) -> bytes:
    header = json.dumps(
        {"data": {"dtype": dtype, "shape": shape, "data_offsets": [0, len(tensor_data)]}},
        separators=(",", ":"),
    ).encode()
    pad = (8 - len(header) % 8) % 8
    header += b" " * pad
    return struct.pack("<Q", len(header)) + header + tensor_data


def write_blob(blobs_dir: str, raw_blob: bytes) -> tuple[str, int, bool]:
    digest = hashlib.sha256(raw_blob).hexdigest()
    blob_path = Path(blobs_dir) / f"sha256-{digest}"
    if blob_path.exists():
        return f"sha256:{digest}", len(raw_blob), False
    temp_path = blob_path.with_suffix(f"{blob_path.suffix}.tmp")
    with temp_path.open("wb") as handle:
        handle.write(raw_blob)
    temp_path.replace(blob_path)
    return f"sha256:{digest}", len(raw_blob), True


def _read_tensor_array(
    path: str, header: dict[str, object], data_offset: int, key: str
) -> np.ndarray:
    info = cast(dict[str, Any], header[key])
    dtype = cast(str, info["dtype"])
    shape = cast(list[int], info["shape"])
    raw = read_tensor_raw(path, header, key, data_offset)
    return to_float32(raw, dtype, shape)


def _as_scalar(arr: np.ndarray, key: str) -> float:
    if arr.size != 1:
        raise ValueError(f"LoRA alpha tensor must contain one value: {key}")
    return float(arr.reshape(-1)[0])


def _compute_delta(
    up: np.ndarray, down: np.ndarray, alpha: float, user_weight: float, key: str
) -> np.ndarray:
    if up.ndim != 2 or down.ndim != 2:
        raise ValueError(f"Only 2D LoRA matrices are supported: {key}")
    if up.shape[1] != down.shape[0]:
        raise ValueError(f"LoRA rank mismatch for {key}: up={up.shape}, down={down.shape}")
    rank = down.shape[0]
    alpha_scale = alpha / rank if rank else 1.0
    return cast(np.ndarray, (up @ down) * float(user_weight * alpha_scale))


def load_lora_deltas(loras: list[LoRASpec]) -> dict[str, np.ndarray]:
    by_target: dict[str, dict[str, np.ndarray]] = defaultdict(dict)
    for spec in loras:
        header, data_offset = read_safetensors_header(spec.path)
        for key in header:
            parsed = parse_lora_tensor_key(key)
            if parsed is None:
                continue
            target_key, part = parsed
            by_target[target_key][part] = _read_tensor_array(spec.path, header, data_offset, key)
        for target_key, part_map in by_target.items():
            down = part_map.get("down")
            up = part_map.get("up")
            if (down is None) != (up is None):
                raise ValueError(f"LoRA target missing up/down pair: {target_key}")

    deltas: dict[str, np.ndarray] = {}
    for spec in loras:
        header, data_offset = read_safetensors_header(spec.path)
        spec_targets: dict[str, dict[str, np.ndarray]] = defaultdict(dict)
        for key in header:
            parsed = parse_lora_tensor_key(key)
            if parsed is None:
                continue
            target_key, part = parsed
            spec_targets[target_key][part] = _read_tensor_array(spec.path, header, data_offset, key)
        for target_key, part_map in spec_targets.items():
            down = part_map.get("down")
            up = part_map.get("up")
            if down is None or up is None:
                continue
            alpha = (
                _as_scalar(part_map["alpha"], target_key)
                if "alpha" in part_map
                else float(down.shape[0])
            )
            delta = _compute_delta(up, down, alpha, spec.user_weight, target_key)
            if target_key in deltas:
                deltas[target_key] = deltas[target_key] + delta
            else:
                deltas[target_key] = delta
    return deltas
