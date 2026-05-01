# AGENTS.md

This file guides LLM agents working in this repository.

## Project Context

`ollama-zit-import` is a CLI that creates local Ollama models from z-image-turbo
weights. It supports two core workflows:

- Full checkpoint import from a z-image-turbo `.safetensors` file
- LoRA merge import that applies one or more adapters to an existing base model

The output is a new named Ollama model reference (for example
`my/imported-zit-model:latest`) that can be run with Ollama. The tool does not use
Ollama's HTTP import API. It writes blobs and manifest files directly in Ollama's model
store layout.

Important operational behavior:

- Preserve non-transformer base layers and keep `transformer/config.json` from base
- Generate or merge transformer weights for output model layers
- Fail if output manifest already exists
- Keep `--dry-run` side-effect free

## Project Snapshot

- Package: `ollama-zit-import`
- Language: Python `>=3.12`
- Layout:
  - Source: `src/ollama_zit_import/`
  - Tests: `tests/`
  - Build config: `pyproject.toml`
  - Task runner: `Makefile`
- Entry points:
  - CLI script: `ollama-zit-import`
  - Module entry: `python3 -m ollama_zit_import`

## Primary Development Tools

- Packaging/build: `setuptools`, `wheel`, `build`
- Testing: `pytest`, `pytest-cov`, `pytest-mock`
- Linting/imports: `ruff`
- Formatting check: `black --check`
- Type checking: `mypy` (strict mode in `pyproject.toml`)
- CLI output: `rich`
- Runtime tensor work: `numpy`

## Standard Commands

Use `make` targets when possible:

- `make dev-install` - editable install with dev dependencies via `uv` into `.venv`
- `make test` - run tests
- `make lint` - run `ruff check .`
- `make format` - run `black --check .`
- `make typecheck` - run `mypy src tests`
- `make check` - run lint + format + typecheck + tests
- `make run-help` - verify module entrypoint help

If you run commands directly, prefer `uv run ...`.

## Agent Workflow

1. Read relevant files first. Start with `README.md`, `technical.md`, this file (including [References](#references)), and touched modules.
2. Keep changes minimal and local. Match existing patterns in `importer.py`, `planning.py`, and tests.
3. Add or update tests for behavior changes in `tests/`.
4. Run validation before finishing:
   - `make lint`
   - `make format`
   - `make typecheck`
   - `make test`
   - or `make check`
5. Do not change unrelated files.

## Code Style Expectations

- Keep functions focused and explicit.
- Use type hints. The repo enforces strict `mypy` settings.
- Avoid magic constants when they are reused. Name them near usage or module top.
- Fail fast for programmer errors. Raise clear exceptions.
- Preserve CLI behavior and user-facing text unless change is intentional.

## Testing Expectations

- Favor unit tests under `tests/test_*.py`.
- Cover both success and failure paths.
- For CLI tests, follow existing monkeypatch + temp directory patterns in `tests/test_cli.py`.
- For parsing and mapping logic, follow focused pattern tests like `tests/test_key_mapping.py` and `tests/test_model_ref.py`.

## Safety Notes

- This tool writes Ollama manifests and blobs. Be careful with file operations.
- Apply thorough due diligence before every write. Only touch files for the specific
  target model being imported. Never modify base images, unrelated models, Ollama
  application configuration, or Ollama internal files outside the scoped import
  artifacts.
- Respect `--dry-run` behavior. Dry run must not write new output manifests or blobs.
- Do not overwrite an existing output manifest path.

## References

Curated links for the model family, weight formats, Ollama’s import and on-disk layout, and adapter theory. Use these when debugging tensor maps, manifests, or merge behavior.

### Z-Image (model, weights, upstream code)

- [Tongyi-MAI/Z-Image-Turbo · Hugging Face](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo) — official weights, cards, and ecosystem notes.
- [Tongyi-MAI/Z-Image · GitHub](https://github.com/Tongyi-MAI/Z-Image) — upstream training/inference code and configuration.
- [Z-Image: An Efficient Image Generation Foundation Model with Single-Stream Diffusion Transformer · arXiv](https://arxiv.org/abs/2511.22699) — architecture and training context (S3-DiT).

### Diffusers and checkpoint semantics

- [Z-Image pipelines · Diffusers documentation](https://huggingface.co/docs/diffusers/api/pipelines/z_image) — `ZImagePipeline` and related APIs; useful for naming and component boundaries versus `.safetensors` dumps.
- [Z-Image pipeline source · huggingface/diffusers](https://github.com/huggingface/diffusers/tree/main/src/diffusers/pipelines/z_image) — ground truth for how the reference implementation loads and runs the model.

### Safetensors

- [Safetensors · Hugging Face documentation](https://huggingface.co/docs/safetensors/index) — header layout, dtypes, and safe loading; aligns with how checkpoints are parsed here.

### LoRA and merging (theory and practice)

- [LoRA: Low-Rank Adaptation of Large Language Models · arXiv:2106.09685](https://arxiv.org/abs/2106.09685) — original formulation of low-rank deltas merged into base weights.
- [Model merging · PEFT documentation](https://huggingface.co/docs/peft/developer_guides/model_merging) — merge strategies and terminology used across the HF stack.

### Ollama (import, Modelfile, API, storage)

- [Importing a model · Ollama docs](https://docs.ollama.com/import) — supported flows and expectations for local weights.
- [Modelfile reference · Ollama docs](https://docs.ollama.com/modelfile) — `FROM`, adapters, and how Ollama names layers in manifests.
- [API documentation · ollama/ollama](https://github.com/ollama/ollama/blob/main/docs/api.md) — blob upload and model creation endpoints (contrast with this CLI’s direct manifest write).
- [Inside Ollama’s model storage — blobs and manifests · Medium](https://medium.com/@dewasheesh.rana/inside-ollamas-model-storage-understanding-blobs-and-manifests-06f1620dd0b2) — unofficial but concrete walkthrough of `manifests/` and `blobs/` layout.

### Quantization and GGUF ecosystem (context for quantized base layers)

- [Difference in different quantization methods · llama.cpp discussion](https://github.com/ggml-org/llama.cpp/discussions/2094) — practical overview of GGUF quant types and trade-offs.
- [Model quantization: concepts and methods · NVIDIA Technical Blog](https://developer.nvidia.com/blog/model-quantization-concepts-methods-and-why-it-matters/) — background on why quantized weights use scales and block structure.

### Community discussion (signal, not authority)

- [Z-Image Turbo — preliminary testing thread · Hacker News](https://news.ycombinator.com/item?id=46175068) — user reports on quality, speed, and tooling around release time.

## Definition of Done

A change is ready when:

- Behavior is implemented and tested.
- `make check` passes locally.
- Test coverage is >=80%
- Documentation is updated if CLI behavior or workflow changed.
