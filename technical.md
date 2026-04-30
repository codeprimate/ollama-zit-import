# ollama-zit-import — technical reference

This document describes how the importer works: assumptions about Ollama’s on-disk layout, how a base manifest is reused, how `.safetensors` weights are read and rewritten, and what the CLI does end-to-end.

## Purpose

`ollama-zit-import` is a Python CLI with two modes:

1. **Standard import** from a z-image-turbo-compatible checkpoint.
2. **LoRA merge import** from an existing local Ollama base model and one or more LoRA adapters.

It does **not** call Ollama’s HTTP API for import itself; it writes **blob files** and a **manifest JSON** in the same layout Ollama uses under `blobs/` and `manifests/registry.ollama.ai/`.

In both modes, the workflow keeps non-transformer layers (VAE, text encoder, tokenizer assets, etc.) from an existing pulled base model, keeps `transformer/config.json`, and writes a new manifest under the user-chosen model name.

## High-level flow

1. **Validate branch inputs**:
   - Standard import: checkpoint file must exist and end with `.safetensors`.
   - LoRA merge import: each `--lora PATH@WEIGHT` file must exist and parse.
2. **Validate models store** — Ollama models root must exist and contain `blobs/`.
3. **Resolve Ollama binary** — `PATH`, then common macOS paths, or `--ollama-bin`. `ollama --version` must succeed.
4. **Ensure base model** — If the base model’s manifest or any referenced blob is missing, run `ollama pull <base>`.
5. **Load base manifest** — Read JSON; retain every layer whose `name` does **not** start with `transformer/` except always retain `transformer/config.json` (required; missing → error).
6. **Mode-specific transformer generation**:
   - Standard import: map and convert checkpoint tensors into transformer layers.
   - LoRA merge import: load LoRA deltas, apply to base transformer tensors, and write merged transformer layers.
   - For quantized base layers (`U32`), read companion `*.weight_scale` and `*.weight_qbias`, dequantize to float32, apply deltas, then repack to `U32`.
7. **Write output manifest** — Same `schemaVersion`, `mediaType`, and `config` as base; `layers` = kept base layers + transformer layers.

Dry run (`--dry-run`) performs planning output and exits **before** creating the output manifest directory or writing any new blobs.

## On-disk layout (assumptions)

| Path (under models root) | Role |
|--------------------------|------|
| `blobs/sha256-<64-hex>` | Content-addressed layer blobs |
| `manifests/registry.ollama.ai/<namespace>/<model>/<tag>` | Manifest JSON (file path mirrors `namespace/name/tag`) |

Models root defaults to `$OLLAMA_MODELS` if set, else `~/.ollama/models`. Override with `--ollama-models`.

## Model reference parsing

User-facing names follow Ollama-style references:

- Optional `namespace/name` — if no `/`, namespace defaults to **`my`**.
- Optional tag after `:` — if omitted, tag is **`latest`**. A trailing lone `:` normalizes to `latest`.

`display_name` is used for `ollama pull` and user messages; `run_name` is suggested for `ollama run` (same as core path when tag is `latest`, else includes `:tag`).

## Base model contract

The base model (default `x/z-image-turbo:latest`) must provide a manifest whose `layers` include **`transformer/config.json`**. That layer’s blob is **not** regenerated from the user checkpoint; it is copied forward by reference (same digest/size entry as in the filtered base list).

All other layers with `name` starting with `transformer/` are **dropped** from the base list before merge; every transformer weight in the output comes from the imported `.safetensors` (except `transformer/config.json`).

## Safetensors I/O

- **Header** — First 8 bytes: little-endian `uint64` header length; next `header_len` bytes: UTF-8 JSON object mapping tensor names to `{dtype, shape, data_offsets}`.
- **Payload** — Tensor bytes start at `8 + header_len + data_offsets[0]` (implementation uses a single `data_offset` for the file’s tensor region).

### Supported dtypes (source)

`BF16`, `F16`, and `F32` are converted to float32 in memory when conversion or QKV splitting is needed. Output tensor blobs are always written as **BF16** in the generated per-tensor safetensors wrapper.

### Per-tensor blob format written to Ollama

Each new layer blob is a minimal safetensors file: length-prefixed JSON header describing one tensor named `"data"` with dtype `BF16` and `data_offsets` covering the raw BF16 bytes, 8-byte aligned header padding, then payload. This matches Ollama’s expectation of `application/vnd.ollama.image.tensor` layers as addressable tensor blobs.

### Deduplication

`write_blob` computes `sha256` over the full blob bytes. If `blobs/sha256-<digest>` already exists, the digest is reused and counters report **reused** vs **written**.

## Weight name mapping and LoRA key analysis

The checkpoint is assumed to use keys prefixed with `model.diffusion_model.` (typical ComfyUI / diffusion naming). That prefix is stripped, then:

| Source pattern | Mapping |
|----------------|---------|
| `final_layer.*` | Prefixed to `all_final_layer.2-1.*` |
| `x_embedder.*` | Prefixed to `all_x_embedder.2-1.*` |
| `*.attention.k_norm.weight` | `*.attention.norm_k.weight` |
| `*.attention.q_norm.weight` | `*.attention.norm_q.weight` |
| `*.attention.out.weight` | `*.attention.to_out.0.weight` |
| `*.attention.qkv.weight` | Split into three logical tensors: `to_q`, `to_k`, `to_v` (first dim divided by 3) |

Standard import non-QKV tensors: if already `BF16`, bytes are passed through; otherwise converted via float32 to BF16.

QKV tensors: loaded as float32, split along the leading dimension into Q/K/V chunks, then each chunk written as its own layer with the names above.

Ollama layer `name` fields are `transformer/` + mapped name (e.g. `transformer/block1.attention.to_q.weight`).

For LoRA mode, adapter keys are parsed from supported suffixes:

- `.lora_down.weight`
- `.lora_up.weight`
- `.lora_A.weight`
- `.lora_B.weight`
- `.alpha`

Keys are normalized and compared to base transformer layer names during dry-run and execution.

## CLI surface

Entry points: `python3 -m ollama_zit_import` or console script `ollama-zit-import` → `ollama_zit_import.__main__:main`, which wraps `cli.run()` and maps uncaught exceptions to exit code 1 and stderr message.

| Argument | Description |
|----------|-------------|
| `safetensors_path` | Input `.safetensors` file (standard mode only) |
| `output_model` | Target Ollama model ref (`namespace/name:tag` or shortcuts per above) |
| `--base-model` | Base manifest to merge LoRA adapters into (required in LoRA mode; default in standard mode) |
| `--lora` | LoRA adapter in `PATH@WEIGHT` form (repeatable; LoRA mode) |
| `--ollama-models` | Models store root |
| `--ollama-bin` | Explicit `ollama` executable |
| `--dry-run` | Validate and print plan only |

## Dependencies

- **numpy** — Buffer dtype conversions, QKV splits, BF16 packing.
- **rich** — Console output, progress bar during tensor conversion.

Python **3.10+**.

## Development tooling

The repository uses `uv` with a project-local `.venv` for developer commands:

```bash
git clone https://github.com/codeprimate/ollama-zit-import.git
cd ollama-zit-import
```

- `make dev-install` installs editable package + dev dependencies into `.venv`
- `make test`, `make lint`, `make format`, and `make typecheck` execute via `uv run ...`
- `make install` builds with `python -m build` and installs the wheel with the active Python environment

Use `uv sync --extra dev` to align the environment with project dependencies. This also
creates `.venv` automatically if it does not exist.

For end users, installation is via `pip` from the GitHub repository:

`python3 -m pip install git+https://github.com/codeprimate/ollama-zit-import.git`

## Limitations and operational notes

- **Compatibility** — Source checkpoints must match the naming and tensor layout expected by the mapping and by the retained `transformer/config.json` from the base Ollama image. Arbitrary diffusion checkpoints may fail or run incorrectly in Ollama even if import succeeds.
- **LoRA coverage** — Supports common `lora_unet_`/`lora_te_` style keys with down/up/alpha tensors, including `.lora_A.weight`/`.lora_B.weight` variants. Unknown patterns are counted and reported.
- **Quantized tensors** — LoRA merging supports quantized `U32` transformer weights only when matching `weight_scale` and `weight_qbias` layers are present in the base model.
- **Side effects** — Normal import mutates the shared blobs directory and adds a manifest; concurrent imports to the same store are not specially coordinated.
- **Registry host** — Manifests are always written under `manifests/registry.ollama.ai/`, matching Ollama’s default registry layout for local images.
- **Collision behavior** — Output manifest path collisions fail fast by default.
- **Errors** — Common failures: missing/invalid safetensors or LoRA files, missing Ollama dirs, `ollama pull` failure, base manifest missing `transformer/config.json`, QKV tensor with leading dim not divisible by 3, unsupported dtype strings, LoRA shape mismatch.

## Related files

| File | Responsibility |
|------|----------------|
| `src/ollama_zit_import/cli.py` | Branch orchestration, Ollama layout, `run()` |
| `src/ollama_zit_import/planning.py` | Mode resolution, model reference parsing, import plan creation |
| `src/ollama_zit_import/lora.py` | LoRA spec parsing and header/key analysis |
| `src/ollama_zit_import/key_mapping.py` | Name normalization and target matching helpers |
| `src/ollama_zit_import/derive.py` | Tensor decode/encode, LoRA delta load/apply, blob write helpers |
| `src/ollama_zit_import/__main__.py` | Top-level exception handler and exit codes |
| `tests/test_model_ref.py` | `parse_model_ref`, `map_name`, dtype helpers |
| `tests/test_lora.py` | LoRA parsing and key analysis coverage |
| `tests/test_key_mapping.py` | base/LoRA target normalization coverage |
| `tests/test_cli.py` | Branch behavior, dry-run reporting, and execution path checks |
