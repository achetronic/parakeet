# Regenerating the ONNX models

`training/export.py` rebuilds every ONNX asset the server loads, from the
official NVIDIA checkpoint. No third-party conversions needed.

## What it produces

Written to `training/assets/`:

| file | what |
| --- | --- |
| `config.json` | model config |
| `vocab.txt` | SentencePiece vocab + `<blk>` blank, 8193 lines |
| `nemo128.onnx` | mel preprocessor, test reference only |
| `encoder-model.onnx` + `.data` | fp32 encoder, full attention |
| `encoder-model.int8.onnx` | int8 encoder, full attention |
| `encoder-local.onnx` + `.data` | fp32 encoder, local attention (`rel_pos_local_attn`) |
| `encoder-local.int8.onnx` | int8 encoder, local attention |
| `decoder_joint-model.onnx` | fp32 decoder + joint |
| `decoder_joint-model.int8.onnx` | int8 decoder + joint |
| `manifest.txt` | sha256 of every artifact |

## Requirements

- Python 3.12. 3.14 does not work with torch/nemo.
- `uv` to build the venv.
- ~20 GB disk. Checkpoint 2.5 GB plus fp32 and int8 assets.

## Setup, once

```sh
make model-training-venv
```

This builds `training/venv` with Python 3.12, torch (cu126) and nemo_toolkit 3.0.

## Generate the assets

```sh
make model-training
```

If the checkpoint is missing it downloads it from
`nvidia/parakeet-tdt-0.6b-v3` (no token required).

Manual run, if you prefer:

```sh
training/venv/bin/python training/export.py \
  --checkpoint training/checkpoint/parakeet-tdt-0.6b-v3.nemo \
  --out training/assets
```

## Local attention: the patch

NeMo 3.0 exports `RelPositionMultiHeadAttentionLongformer` wrong. Its
sliding-chunks path rebuilds the attention band with `new_empty()` plus
in-place slice assignments. The torchscript exporter mis-compiles those and
the encoder outputs garbage.

`export.py` replaces that forward with a full-attention + band-mask version.
Numerically equivalent (max diff ~1e-3) and uses only export-safe ops.

Caveat: the replacement is O(T²) in memory, not O(T·w). Fine for audio up to
tens of minutes. For hours, fix the sliding-chunks export instead.

## nemo128.onnx

Test reference only. The server never loads it; `internal/mel` handles the mel
frontend. torch>=2.x makes `torch.stft` return complex values, which
torchscript cannot trace, so the export is best-effort and usually skipped.
To regenerate it, patch the preprocessor stft to `return_complex=False`.

## Publishing to HuggingFace

Assets are gitignored. They go to your own HuggingFace repo and the Makefile
downloads them from there.

```sh
huggingface-cli upload <your-user>/parakeet-tdt-0.6b-v3-onnx training/assets/ . --repo-type model
```

Then point the Makefile at it:

```sh
make MODELS_REPO=<your-user>/parakeet-tdt-0.6b-v3-onnx models-int8
```
