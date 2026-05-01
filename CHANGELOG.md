# Changelog

All notable changes to this project are documented in this file.

The format is inspired by [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [0.3.1] - 2026-05-01

### Added

- After a successful import, the CLI prints a **Store** line: IEC sizes (KiB–TiB) for new blob data and for the new manifest, with a short note that shared digests are not counted.
- README **Quick Start**: install via `pip` from Git, minimal checkpoint import, and LoRA merge examples.
- README **References**: curated links for Z-Image, Diffusers, Safetensors, LoRA merging, Ollama import and storage, quantization context, and related discussion.

### Changed

- `execute_standard_import` and `execute_lora_derivation` now return the total byte size of blobs newly written during the run (fourth integer before warnings in the LoRA path).

### Tests

- `tests/test_importer.py` covers IEC byte formatting; CLI and model importer tests updated for the new summary and return values.
