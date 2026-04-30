"""Tensor key normalization helpers."""

from __future__ import annotations

from typing import Iterable


def normalize_base_transformer_name(layer_name: str) -> str:
    return layer_name.removeprefix("transformer/")


def count_matching_targets(base_layer_names: Iterable[str], lora_target_keys: Iterable[str]) -> int:
    normalized_base = {normalize_base_transformer_name(name) for name in base_layer_names}
    return sum(1 for target_key in lora_target_keys if target_key in normalized_base)


def map_name(comfy_name: str) -> list[tuple[str, str | None]]:
    name = comfy_name.removeprefix("model.diffusion_model.")
    if name.startswith("final_layer."):
        name = "all_final_layer.2-1." + name[len("final_layer.") :]
    elif name.startswith("x_embedder."):
        name = "all_x_embedder.2-1." + name[len("x_embedder.") :]

    name = name.replace(".attention.k_norm.weight", ".attention.norm_k.weight")
    name = name.replace(".attention.q_norm.weight", ".attention.norm_q.weight")
    name = name.replace(".attention.out.weight", ".attention.to_out.0.weight")

    if name.endswith(".attention.qkv.weight"):
        prefix = name[: -len(".attention.qkv.weight")]
        return [
            (f"{prefix}.attention.to_q.weight", "q"),
            (f"{prefix}.attention.to_k.weight", "k"),
            (f"{prefix}.attention.to_v.weight", "v"),
        ]
    return [(name, None)]

