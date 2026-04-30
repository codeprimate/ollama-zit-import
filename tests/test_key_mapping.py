"""Key mapping utility tests."""

from __future__ import annotations

import pytest

from ollama_zit_import.key_mapping import count_matching_targets


@pytest.mark.unit
def test_count_matching_targets_uses_normalized_transformer_names() -> None:
    base_layers = [
        "transformer/down.blocks.0.attentions.0.to_q.weight",
        "transformer/down.blocks.0.attentions.0.to_k.weight",
    ]
    lora_targets = {
        "down.blocks.0.attentions.0.to_q.weight",
        "down.blocks.0.attentions.0.to_v.weight",
    }
    assert count_matching_targets(base_layers, lora_targets) == 1
