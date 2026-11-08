# Regenerating the ONNX models

`export/asr/export.py` rebuilds every ASR ONNX asset the server loads, from
the official NVIDIA checkpoint. No third-party conversions needed.

The directory is split by model family so future exports (VAD, diarization)
get their own subfolder. Each family keeps its code, checkpoint, requirements
and assets inside its own `export/<family>/` directory. The Makefile has
per-family targets, so `asr` is explicit in the target names.

```
export/
├── asr/               export.py + Dockerfile.models + build-image.sh
│   ├── checkpoint/    .nemo checkpoint, gitignored
│   ├── assets/        generated output, gitignored
│   └── requirements.txt
├── venv/              shared export environment, gitignored
└── README.md
```

## What the export produces

Written to `export/asr/assets/`:

| file                            | what                                                 |
| ------------------------------- | ---------------------------------------------------- |
| `config.json`                   | model config                                         |
| `vocab.txt`                     | SentencePiece vocab + `<blk>` blank, 8193 lines      |
| `nemo128.onnx`                  | mel preprocessor, test reference only                |
| `encoder-model.onnx` + `.data`  | fp32 encoder, local attention (`rel_pos_local_attn`) |
| `encoder-model.int8.onnx`       | int8 encoder, local attention                        |
| `decoder_joint-model.onnx`      | fp32 decoder + joint                                 |
| `decoder_joint-model.int8.onnx` | int8 decoder + joint                                 |
| `manifest.txt`                  | sha256 of every artifact                             |

## Requirements

- Python 3.12. 3.14 does not work with torch/nemo.
- `uv` to build the venv.
- ~20 GB disk. Checkpoint 2.5 GB plus fp32 and int8 assets.
- Docker, to build the model images.

## Setup, once

```sh
make model-export-asr-venv
```

This builds `export/venv` with Python 3.12 and torch, plus the dependencies
from `export/asr/requirements.txt` (nemo_toolkit).

## Regenerate the assets

```sh
make model-export-asr
```

If the checkpoint is missing it downloads it from
`nvidia/parakeet-tdt-0.6b-v3` (no token required) into
`export/asr/checkpoint/`.

Manual run:

```sh
export/venv/bin/python export/asr/export.py
```

`--checkpoint` and `--out` default to `export/asr/checkpoint/` and
`export/asr/assets/`.

## Model images

The assets are shipped as OCI images on GitHub Container Registry. Two images,
one per precision, each with the local-attention encoder and the decoder:

| image                                     | contents                        |
| ----------------------------------------- | ------------------------------- |
| `ghcr.io/achetronic/parakeet-models-asr-int8` | int8 encoder + decoder + vocab |
| `ghcr.io/achetronic/parakeet-models-asr-fp32` | fp32 encoder + decoder + vocab |

Build locally without pushing:

```sh
make model-build-oci-images
```

Build and push:

```sh
docker login ghcr.io -u <user> --password-stdin   # once, with a PAT
make MODEL_TAG=v1.0.0 model-push-oci-images
```

`MODEL_TAG` defaults to `latest`.

## CI regeneration

`.github/workflows/export-models.yaml` regenerates everything and pushes both images.
Trigger it manually with a tag. Empty tag means checkout `main` and publish
`:latest`; a tag checks out that ref and publishes under that tag.

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
