"""CLI behavior tests."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import ollama_zit_import.cli as cli_module
from ollama_zit_import.cli import run
from ollama_zit_import.derive import make_single_tensor_blob


def _write_safetensors(path: Path, header: dict[str, object], payload: bytes = b"\x00\x00") -> None:
    header_raw = json.dumps(header).encode("utf-8")
    path.write_bytes(len(header_raw).to_bytes(8, "little") + header_raw + payload)


@pytest.mark.unit
def test_main_help_text() -> None:
    from subprocess import run as sp_run

    env = dict(os.environ)
    env["PYTHONPATH"] = str(Path("src").resolve())
    proc = sp_run(
        [sys.executable, "-m", "ollama_zit_import", "--help"],
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )
    assert proc.returncode == 0
    assert "Import z-image-turbo safetensors into Ollama as a new model." in proc.stdout


@pytest.mark.unit
def test_dry_run_path(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    safetensors = tmp_path / "mini.safetensors"
    _write_safetensors(
        safetensors,
        {
            "__metadata__": {},
            "model.diffusion_model.foo.weight": {
                "dtype": "BF16",
                "shape": [1],
                "data_offsets": [0, 2],
            },
        },
    )

    models_root = tmp_path / "models"
    blobs_dir = models_root / "blobs"
    manifest_dir = models_root / "manifests" / "registry.ollama.ai" / "x" / "z-image-turbo"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    blobs_dir.mkdir(parents=True, exist_ok=True)

    base_manifest = manifest_dir / "latest"
    base_manifest.write_text(
        json.dumps(
            {
                "schemaVersion": 2,
                "mediaType": "application/vnd.ollama.image.manifest",
                "config": {"mediaType": "application/vnd.ollama.image.config"},
                "layers": [
                    {"name": "transformer/config.json", "digest": "sha256:abcd", "size": 12}
                ],
            }
        ),
        encoding="utf-8",
    )
    (blobs_dir / "sha256-abcd").write_bytes(b"blob")

    monkeypatch.setattr(
        "ollama_zit_import.cli.parse_args",
        lambda: SimpleNamespace(
            safetensors_path=str(safetensors),
            output_model="x/new-model:latest",
            base_model="x/z-image-turbo:latest",
            ollama_models=str(models_root),
            ollama_bin="/usr/local/bin/ollama",
            dry_run=True,
            lora=None,
        ),
    )
    monkeypatch.setattr("ollama_zit_import.runtime_support.check_ollama_binary", lambda _p: None)
    monkeypatch.setattr(
        "ollama_zit_import.runtime_support.detect_ollama_binary", lambda explicit: str(explicit)
    )

    assert run() == 0


@pytest.mark.unit
def test_lora_dry_run_reports_match_stats(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    lora_file = tmp_path / "adapter.safetensors"
    _write_safetensors(
        lora_file,
        {
            "__metadata__": {},
            "lora_unet_down_blocks_0_attentions_0_to_q.lora_down.weight": {
                "dtype": "F16",
                "shape": [1],
                "data_offsets": [0, 2],
            },
            "noise.key": {"dtype": "F16", "shape": [1], "data_offsets": [0, 2]},
        },
    )
    models_root = tmp_path / "models"
    blobs_dir = models_root / "blobs"
    manifest_dir = models_root / "manifests" / "registry.ollama.ai" / "x" / "base"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    blobs_dir.mkdir(parents=True, exist_ok=True)
    base_manifest = manifest_dir / "latest"
    base_manifest.write_text(
        json.dumps(
            {
                "schemaVersion": 2,
                "mediaType": "application/vnd.ollama.image.manifest",
                "config": {"mediaType": "application/vnd.ollama.image.config"},
                "layers": [
                    {"name": "transformer/config.json", "digest": "sha256:abcd", "size": 12},
                    {
                        "name": "transformer/down.blocks.0.attentions.0.to.q.weight",
                        "digest": "sha256:efgh",
                        "size": 8,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    (blobs_dir / "sha256-abcd").write_bytes(b"blob")
    (blobs_dir / "sha256-efgh").write_bytes(b"blob")
    monkeypatch.setattr(
        "ollama_zit_import.cli.parse_args",
        lambda: SimpleNamespace(
            output_model="x/new-model:latest",
            base_model="x/base:latest",
            lora=[f"{lora_file}@0.5"],
            ollama_models=str(models_root),
            ollama_bin="/usr/local/bin/ollama",
            dry_run=True,
        ),
    )
    monkeypatch.setattr("ollama_zit_import.runtime_support.check_ollama_binary", lambda _p: None)
    monkeypatch.setattr(
        "ollama_zit_import.runtime_support.detect_ollama_binary", lambda explicit: str(explicit)
    )
    assert run() == 0
    captured = capsys.readouterr().out
    assert "branch=lora_only_derivation" in captured
    assert "matched_keys=1" in captured
    assert "unmatched_keys=1" in captured


@pytest.mark.unit
def test_lora_execution_writes_output_manifest(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    lora_file = tmp_path / "adapter.safetensors"
    _write_safetensors(
        lora_file,
        {
            "__metadata__": {},
            "lora_unet_down_blocks_0_attentions_0_to_q.weight.lora_down.weight": {
                "dtype": "F32",
                "shape": [1, 2],
                "data_offsets": [0, 8],
            },
            "lora_unet_down_blocks_0_attentions_0_to_q.weight.lora_up.weight": {
                "dtype": "F32",
                "shape": [2, 1],
                "data_offsets": [8, 16],
            },
        },
        payload=b"\x00\x00\x80?\x00\x00\x80?\x00\x00\x00@\x00\x00@@",
    )

    models_root = tmp_path / "models"
    blobs_dir = models_root / "blobs"
    manifest_dir = models_root / "manifests" / "registry.ollama.ai" / "x" / "base"
    output_manifest = (
        models_root / "manifests" / "registry.ollama.ai" / "x" / "new-model" / "latest"
    )
    manifest_dir.mkdir(parents=True, exist_ok=True)
    blobs_dir.mkdir(parents=True, exist_ok=True)

    base_tensor = b"\x00\x00\x80?\x00\x00\x00@\x00\x00@@\x00\x00\x80@"
    base_blob = make_single_tensor_blob("F32", [2, 2], base_tensor)
    base_digest = "sha256:baseblob"
    (blobs_dir / "sha256-baseblob").write_bytes(base_blob)
    (blobs_dir / "sha256-abcd").write_bytes(b"blob")

    base_manifest = manifest_dir / "latest"
    base_manifest.write_text(
        json.dumps(
            {
                "schemaVersion": 2,
                "mediaType": "application/vnd.ollama.image.manifest",
                "config": {"mediaType": "application/vnd.ollama.image.config"},
                "layers": [
                    {"name": "transformer/config.json", "digest": "sha256:abcd", "size": 12},
                    {
                        "name": "transformer/down.blocks.0.attentions.0.to.q.weight",
                        "digest": base_digest,
                        "size": len(base_blob),
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "ollama_zit_import.cli.parse_args",
        lambda: SimpleNamespace(
            output_model="x/new-model:latest",
            base_model="x/base:latest",
            lora=[f"{lora_file}@0.5"],
            ollama_models=str(models_root),
            ollama_bin="/usr/local/bin/ollama",
            dry_run=False,
        ),
    )
    monkeypatch.setattr("ollama_zit_import.runtime_support.check_ollama_binary", lambda _p: None)
    monkeypatch.setattr(
        "ollama_zit_import.runtime_support.detect_ollama_binary", lambda explicit: str(explicit)
    )

    assert run() == 0
    assert output_manifest.exists()


@pytest.mark.unit
def test_lora_execution_fails_when_output_manifest_exists(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    lora_file = tmp_path / "adapter.safetensors"
    _write_safetensors(
        lora_file,
        {
            "__metadata__": {},
            "lora_unet_down_blocks_0_attentions_0_to_q.weight.lora_down.weight": {
                "dtype": "F32",
                "shape": [1, 2],
                "data_offsets": [0, 8],
            },
            "lora_unet_down_blocks_0_attentions_0_to_q.weight.lora_up.weight": {
                "dtype": "F32",
                "shape": [2, 1],
                "data_offsets": [8, 16],
            },
        },
        payload=b"\x00\x00\x80?\x00\x00\x80?\x00\x00\x00@\x00\x00@@",
    )
    models_root = tmp_path / "models"
    blobs_dir = models_root / "blobs"
    manifest_dir = models_root / "manifests" / "registry.ollama.ai" / "x" / "base"
    output_manifest = (
        models_root / "manifests" / "registry.ollama.ai" / "x" / "new-model" / "latest"
    )
    manifest_dir.mkdir(parents=True, exist_ok=True)
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    blobs_dir.mkdir(parents=True, exist_ok=True)
    output_manifest.write_text("{}", encoding="utf-8")

    base_tensor = b"\x00\x00\x80?\x00\x00\x00@\x00\x00@@\x00\x00\x80@"
    base_blob = make_single_tensor_blob("F32", [2, 2], base_tensor)
    (blobs_dir / "sha256-baseblob").write_bytes(base_blob)
    (blobs_dir / "sha256-abcd").write_bytes(b"blob")

    (manifest_dir / "latest").write_text(
        json.dumps(
            {
                "schemaVersion": 2,
                "mediaType": "application/vnd.ollama.image.manifest",
                "config": {"mediaType": "application/vnd.ollama.image.config"},
                "layers": [
                    {"name": "transformer/config.json", "digest": "sha256:abcd", "size": 12},
                    {
                        "name": "transformer/down.blocks.0.attentions.0.to.q.weight",
                        "digest": "sha256:baseblob",
                        "size": len(base_blob),
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "ollama_zit_import.cli.parse_args",
        lambda: SimpleNamespace(
            output_model="x/new-model:latest",
            base_model="x/base:latest",
            lora=[f"{lora_file}@0.5"],
            ollama_models=str(models_root),
            ollama_bin="/usr/local/bin/ollama",
            dry_run=False,
        ),
    )
    monkeypatch.setattr("ollama_zit_import.runtime_support.check_ollama_binary", lambda _p: None)
    monkeypatch.setattr(
        "ollama_zit_import.runtime_support.detect_ollama_binary", lambda explicit: str(explicit)
    )

    with pytest.raises(FileExistsError, match="Output manifest already exists"):
        run()


@pytest.mark.unit
def test_parse_args_lora_branch_requires_base_model(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    lora_file = tmp_path / "adapter.safetensors"
    lora_file.write_bytes(b"lora")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prog",
            "x/out:latest",
            "--lora",
            f"{lora_file}@0.5",
            "--base-model",
            "x/base:latest",
        ],
    )
    parsed = cli_module.parse_args()
    assert parsed.output_model == "x/out:latest"
    assert parsed.base_model == "x/base:latest"
    assert parsed.lora == [f"{lora_file}@0.5"]


@pytest.mark.unit
def test_parse_args_standard_branch_parses_checkpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["prog", "/tmp/model.safetensors", "x/out:latest"])
    parsed = cli_module.parse_args()
    assert parsed.safetensors_path == "/tmp/model.safetensors"
    assert parsed.output_model == "x/out:latest"
