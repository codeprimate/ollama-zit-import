"""Unit tests for standard import execution helpers."""

from __future__ import annotations

from typing import cast

import numpy as np
import pytest

from ollama_zit_import.model_importer import execute_standard_import


@pytest.mark.unit
def test_execute_standard_import_bf16_passthrough_and_callback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_tensor = "model.diffusion_model.example.weight"
    raw_bf16 = b"\x01\x02\x03\x04"
    header: dict[str, object] = {
        source_tensor: {
            "dtype": "BF16",
            "shape": [2, 1],
        }
    }

    calls: dict[str, object] = {"blob_shape": None, "layer_name": None, "completed": 0}

    monkeypatch.setattr(
        "ollama_zit_import.model_importer.read_tensor_raw",
        lambda *_args, **_kwargs: raw_bf16,
    )
    monkeypatch.setattr(
        "ollama_zit_import.model_importer.map_name",
        lambda _name: [("example.weight", None)],
    )

    def fake_make_single_tensor_blob(_dtype: str, out_shape: list[int], bf16_data: bytes) -> bytes:
        calls["blob_shape"] = list(out_shape)
        return bf16_data

    monkeypatch.setattr(
        "ollama_zit_import.model_importer.make_single_tensor_blob",
        fake_make_single_tensor_blob,
    )
    monkeypatch.setattr(
        "ollama_zit_import.model_importer.write_blob",
        lambda _blobs_dir, _blob: ("sha256:new", 4, True),
    )

    def fake_layer_entry(name: str, digest: str, size: int) -> dict[str, object]:
        calls["layer_name"] = name
        return {"name": name, "digest": digest, "size": size}

    monkeypatch.setattr("ollama_zit_import.model_importer.layer_entry", fake_layer_entry)

    def on_complete() -> None:
        completed = cast(int, calls["completed"])
        calls["completed"] = completed + 1

    layers, blobs_new, blobs_reused = execute_standard_import(
        source_tensors=[source_tensor],
        header=header,
        data_offset=0,
        safetensors_path="/tmp/input.safetensors",
        blobs_dir="/tmp/blobs",
        initial_layers=[],
        on_source_tensor_complete=on_complete,
    )

    assert blobs_new == 1
    assert blobs_reused == 0
    assert calls["blob_shape"] == [2, 1]
    assert calls["layer_name"] == "transformer/example.weight"
    assert cast(int, calls["completed"]) == 1
    assert len(layers) == 1


@pytest.mark.unit
def test_execute_standard_import_splits_qkv_and_counts_reused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_tensor = "model.diffusion_model.attn.qkv.weight"
    header: dict[str, object] = {
        source_tensor: {
            "dtype": "F32",
            "shape": [6, 1],
        }
    }
    source_arr = np.arange(6, dtype=np.float32).reshape(6, 1)

    seen_shapes: list[list[int]] = []

    monkeypatch.setattr(
        "ollama_zit_import.model_importer.read_tensor_raw",
        lambda *_args, **_kwargs: b"unused",
    )
    monkeypatch.setattr(
        "ollama_zit_import.model_importer.to_float32",
        lambda *_args, **_kwargs: source_arr,
    )
    monkeypatch.setattr(
        "ollama_zit_import.model_importer.map_name",
        lambda _name: [
            ("attn.to_q.weight", "q"),
            ("attn.to_k.weight", "k"),
            ("attn.to_v.weight", "v"),
        ],
    )
    monkeypatch.setattr(
        "ollama_zit_import.model_importer.to_bf16_bytes",
        lambda arr: arr.tobytes(),
    )

    def fake_make_single_tensor_blob(_dtype: str, shape: list[int], _data: bytes) -> bytes:
        seen_shapes.append(list(shape))
        return b"blob"

    monkeypatch.setattr(
        "ollama_zit_import.model_importer.make_single_tensor_blob",
        fake_make_single_tensor_blob,
    )
    monkeypatch.setattr(
        "ollama_zit_import.model_importer.write_blob",
        lambda _blobs_dir, _blob: ("sha256:reused", 8, False),
    )

    layers, blobs_new, blobs_reused = execute_standard_import(
        source_tensors=[source_tensor],
        header=header,
        data_offset=0,
        safetensors_path="/tmp/input.safetensors",
        blobs_dir="/tmp/blobs",
        initial_layers=[],
    )

    assert blobs_new == 0
    assert blobs_reused == 3
    assert seen_shapes == [[2, 1], [2, 1], [2, 1]]
    assert [str(layer["name"]) for layer in layers] == [
        "transformer/attn.to_q.weight",
        "transformer/attn.to_k.weight",
        "transformer/attn.to_v.weight",
    ]


@pytest.mark.unit
def test_execute_standard_import_rejects_invalid_qkv_shape(monkeypatch: pytest.MonkeyPatch) -> None:
    source_tensor = "model.diffusion_model.attn.qkv.weight"
    header: dict[str, object] = {
        source_tensor: {
            "dtype": "F32",
            "shape": [5, 2],
        }
    }

    monkeypatch.setattr(
        "ollama_zit_import.model_importer.read_tensor_raw",
        lambda *_args, **_kwargs: b"unused",
    )
    monkeypatch.setattr(
        "ollama_zit_import.model_importer.to_float32",
        lambda *_args, **_kwargs: np.zeros((5, 2), dtype=np.float32),
    )
    monkeypatch.setattr(
        "ollama_zit_import.model_importer.map_name",
        lambda _name: [("attn.to_q.weight", "q")],
    )

    with pytest.raises(ValueError, match="QKV tensor has unsupported shape for split"):
        execute_standard_import(
            source_tensors=[source_tensor],
            header=header,
            data_offset=0,
            safetensors_path="/tmp/input.safetensors",
            blobs_dir="/tmp/blobs",
            initial_layers=[],
        )
