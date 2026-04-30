"""LoRA parser unit tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from ollama_zit_import.lora import analyze_lora_header, parse_lora_spec, parse_lora_tensor_key


@pytest.mark.unit
def test_parse_lora_spec_accepts_valid_value(tmp_path: Path) -> None:
    lora_file = tmp_path / "adapter.safetensors"
    lora_file.write_bytes(b"x")
    spec = parse_lora_spec(f"{lora_file}@0.7")
    assert spec.path.endswith("adapter.safetensors")
    assert spec.user_weight == 0.7


@pytest.mark.unit
def test_parse_lora_spec_rejects_missing_separator() -> None:
    with pytest.raises(ValueError, match="PATH@WEIGHT"):
        parse_lora_spec("/tmp/adapter.safetensors")


@pytest.mark.unit
def test_parse_lora_spec_rejects_out_of_range_weight() -> None:
    with pytest.raises(ValueError, match="between 0.1 and 1.0"):
        parse_lora_spec("/tmp/adapter.safetensors@1.1")


@pytest.mark.unit
def test_analyze_lora_header_tracks_recognized_and_unrecognized() -> None:
    analysis = analyze_lora_header(
        {
            "__metadata__": {},
            "lora_unet_down_blocks_0_attentions_0_to_q.lora_down.weight": object(),
            "lora_unet_down_blocks_0_attentions_0_to_q.lora_up.weight": object(),
            "unexpected.tensor": object(),
        }
    )
    assert "down.blocks.0.attentions.0.to.q.weight" in analysis.recognized_target_keys
    assert analysis.unmatched_count == 1


@pytest.mark.unit
def test_parse_lora_tensor_key_supports_diffusion_model_lora_ab() -> None:
    parsed_a = parse_lora_tensor_key("diffusion_model.layers.0.feed_forward.w1.lora_A.weight")
    parsed_b = parse_lora_tensor_key("diffusion_model.layers.0.feed_forward.w1.lora_B.weight")
    assert parsed_a == ("layers.0.feed_forward.w1.weight", "down")
    assert parsed_b == ("layers.0.feed_forward.w1.weight", "up")
