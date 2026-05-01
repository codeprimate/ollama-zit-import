"""Tests for importer orchestration helpers."""

from __future__ import annotations

import pytest

from ollama_zit_import.importer import _format_iec_bytes


@pytest.mark.unit
@pytest.mark.parametrize(
    ("size", "expected"),
    [
        (0, "0 B"),
        (500, "500 B"),
        (1024, "1.0 KiB"),
        (20 * 1024 * 1024, "20.0 MiB"),
        (13286400000, "12.37 GiB"),
    ],
)
def test_format_iec_bytes(size: int, expected: str) -> None:
    assert _format_iec_bytes(size) == expected
