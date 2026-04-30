"""Unit tests for pure helper logic."""

from __future__ import annotations

import numpy as np
import pytest

from ollama_zit_import.derive import quantize_float32_to_u32_packed, to_bf16_bytes, to_float32
from ollama_zit_import.key_mapping import map_name
from ollama_zit_import.planning import parse_model_ref


@pytest.mark.unit
def test_parse_model_ref_defaults_namespace_and_tag() -> None:
    parsed = parse_model_ref("my-model")
    assert parsed.namespace == "my"
    assert parsed.name == "my-model"
    assert parsed.tag == "latest"
    assert parsed.display_name == "my/my-model"


@pytest.mark.unit
def test_parse_model_ref_with_namespace_and_tag() -> None:
    parsed = parse_model_ref("me/model:v2")
    assert parsed.namespace == "me"
    assert parsed.name == "model"
    assert parsed.tag == "v2"
    assert parsed.display_name == "me/model:v2"


@pytest.mark.unit
def test_parse_model_ref_allows_uppercase_namespace_and_name() -> None:
    parsed = parse_model_ref("My/ModelName:latest")
    assert parsed.namespace == "My"
    assert parsed.name == "ModelName"
    assert parsed.tag == "latest"
    assert parsed.display_name == "My/ModelName"


@pytest.mark.unit
def test_parse_model_ref_rejects_empty() -> None:
    with pytest.raises(ValueError):
        parse_model_ref("   ")


@pytest.mark.unit
@pytest.mark.parametrize(
    "model_ref",
    [
        "../evil:latest",
        "ns/../evil:latest",
        "ns/evil/name:latest",
        "ns/evil:../latest",
        "ns/model$:latest",
    ],
)
def test_parse_model_ref_rejects_unsafe_or_invalid_components(model_ref: str) -> None:
    with pytest.raises(ValueError):
        parse_model_ref(model_ref)


@pytest.mark.unit
def test_map_name_splits_qkv() -> None:
    mapped = map_name("model.diffusion_model.block1.attention.qkv.weight")
    assert mapped == [
        ("block1.attention.to_q.weight", "q"),
        ("block1.attention.to_k.weight", "k"),
        ("block1.attention.to_v.weight", "v"),
    ]


@pytest.mark.unit
def test_bf16_roundtrip_shape() -> None:
    arr = np.array([[1.25, -0.5], [3.0, 4.5]], dtype=np.float32)
    bf16 = to_bf16_bytes(arr)
    recovered = to_float32(bf16, "BF16", [2, 2])
    assert recovered.shape == (2, 2)
    assert recovered.dtype == np.float32


@pytest.mark.unit
def test_u32_dequantization_with_scale_and_qbias() -> None:
    packed = np.array([[0x04030201, 0x08070605]], dtype=np.uint32)
    scale = np.array([[2.0]], dtype=np.float32)
    qbias = np.array([[1.0]], dtype=np.float32)
    arr = to_float32(
        packed.tobytes(),
        "U32",
        [1, 2],
        scale=scale,
        qbias=qbias,
        key="transformer/example.weight",
    )
    expected = np.array([[0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0]], dtype=np.float32)
    assert arr.shape == (1, 8)
    assert np.allclose(arr, expected)


@pytest.mark.unit
def test_u32_quantize_roundtrip_with_scale_and_qbias() -> None:
    original = np.array([[0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0]], dtype=np.float32)
    scale = np.array([[2.0]], dtype=np.float32)
    qbias = np.array([[1.0]], dtype=np.float32)
    packed = quantize_float32_to_u32_packed(
        original, scale, qbias, key="transformer/example.weight"
    )
    recovered = to_float32(
        packed.tobytes(),
        "U32",
        list(packed.shape),
        scale=scale,
        qbias=qbias,
        key="transformer/example.weight",
    )
    assert recovered.shape == original.shape
    assert np.allclose(recovered, original)
