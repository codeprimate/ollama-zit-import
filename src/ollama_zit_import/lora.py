"""LoRA argument parsing and validation helpers."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

MIN_LORA_WEIGHT = 0.1
MAX_LORA_WEIGHT = 1.0
LORA_TARGET_SUFFIXES = (
    ".lora_down.weight",
    ".lora_up.weight",
    ".lora_A.weight",
    ".lora_B.weight",
    ".alpha",
)
COMMON_LORA_PREFIXES = ("lora_unet_", "lora_te_")
LORA_PART_FROM_SUFFIX = {
    ".lora_down.weight": "down",
    ".lora_up.weight": "up",
    ".lora_A.weight": "down",
    ".lora_B.weight": "up",
    ".alpha": "alpha",
}
DIFFUSION_MODEL_PREFIX = "diffusion_model."


@dataclass(frozen=True)
class LoRASpec:
    path: str
    user_weight: float


@dataclass(frozen=True)
class LoRAHeaderAnalysis:
    recognized_target_keys: set[str]
    unrecognized_keys: list[str]

    @property
    def matched_count(self) -> int:
        return len(self.recognized_target_keys)

    @property
    def unmatched_count(self) -> int:
        return len(self.unrecognized_keys)


def parse_lora_spec(raw_value: str) -> LoRASpec:
    path_text, separator, weight_text = raw_value.rpartition("@")
    if separator == "":
        raise ValueError(f"Invalid --lora value (expected PATH@WEIGHT): {raw_value}")

    path = path_text.strip()
    if not path:
        raise ValueError(f"LoRA path cannot be empty: {raw_value}")

    try:
        weight = float(weight_text)
    except ValueError as exc:
        raise ValueError(f"LoRA weight is not a number: {raw_value}") from exc

    if weight < MIN_LORA_WEIGHT or weight > MAX_LORA_WEIGHT:
        raise ValueError(
            f"LoRA weight must be between {MIN_LORA_WEIGHT} and {MAX_LORA_WEIGHT}: {raw_value}"
        )

    return LoRASpec(path=str(Path(path).resolve()), user_weight=weight)


def parse_lora_specs(raw_values: list[str]) -> list[LoRASpec]:
    if not raw_values:
        raise ValueError("At least one --lora PATH@WEIGHT value is required")
    return [parse_lora_spec(raw_value) for raw_value in raw_values]


def _strip_known_lora_suffix(key: str) -> tuple[str, str] | None:
    for suffix in LORA_TARGET_SUFFIXES:
        if key.endswith(suffix):
            return key[: -len(suffix)], LORA_PART_FROM_SUFFIX[suffix]
    return None


def _normalize_lora_target(stem: str) -> str:
    normalized = stem
    if normalized.startswith(DIFFUSION_MODEL_PREFIX):
        normalized = normalized[len(DIFFUSION_MODEL_PREFIX) :]
        return normalized
    for prefix in COMMON_LORA_PREFIXES:
        if normalized.startswith(prefix):
            normalized = normalized[len(prefix) :]
            normalized = normalized.replace("_", ".")
            break
    normalized = re.sub(r"\.+", ".", normalized).strip(".")
    return normalized


def analyze_lora_header(header: dict[str, object]) -> LoRAHeaderAnalysis:
    recognized_target_keys: set[str] = set()
    unrecognized_keys: list[str] = []
    for key in header:
        if key == "__metadata__":
            continue
        parsed = parse_lora_tensor_key(key)
        if parsed is None:
            unrecognized_keys.append(key)
            continue
        normalized, _part = parsed
        recognized_target_keys.add(normalized)
    return LoRAHeaderAnalysis(
        recognized_target_keys=recognized_target_keys,
        unrecognized_keys=sorted(unrecognized_keys),
    )


def parse_lora_tensor_key(key: str) -> tuple[str, str] | None:
    parsed = _strip_known_lora_suffix(key)
    if parsed is None:
        return None
    stem, part = parsed
    normalized = _normalize_lora_target(stem)
    if not normalized:
        return None
    if not normalized.endswith(".weight"):
        normalized = f"{normalized}.weight"
    return normalized, part
