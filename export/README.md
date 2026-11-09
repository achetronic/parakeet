# Regenerating the ONNX models

The server loads ONNX assets built here, from the official NVIDIA checkpoints.
No third-party conversions. Two families today: ASR and speaker diarization.

Each family keeps its code, checkpoint, requirements and assets inside its own
`export/<family>/` directory, and the Makefile has per-family targets, so the
family name is explicit in every target.

```
export/
├── asr/                 export.py + Dockerfile.models + build-image.sh
│   ├── checkpoint/      .nemo checkpoint, gitignored
│   ├── assets/          generated output, gitignored
│   └── requirements.txt
├── diarization/         export.py + Dockerfile.models + build-image.sh
│   ├── checkpoint/      safetensors checkpoint, gitignored
│   ├── assets/          generated output, gitignored
│   └── requirements.txt
├── venv/                ASR export environment, gitignored
├── venv-diarization/    diarization export environment, gitignored
└── README.md
```

The two venvs cannot be shared: ASR pins `nemo_toolkit` (transformers 4.x),
diarization needs transformers 5.x.

## ASR

### What the export produces

Written to `export/asr/assets/`:

| file                            | what                                                 |
| ------------------------------- | ---------------------------------------------------- |
| `config.json`                   | model config                                         |
| `vocab.txt`                     | SentencePiece vocab + `<blk>` blank, 8193 lines      |
| `nemo128.onnx`                  | mel preprocessor, test reference only                |
| `encoder-full.onnx` + `.data`  | fp32 encoder, full attention                         |
| `encoder-full.int8.onnx`       | int8 encoder, full attention                         |
| `encoder-local.onnx` + `.data` | fp32 encoder, local attention (`rel_pos_local_attn`) |
| `encoder-local.int8.onnx`      | int8 encoder, local attention                        |
| `decoder_joint-model.onnx`      | fp32 decoder + joint                                 |
| `decoder_joint-model.int8.onnx` | int8 decoder + joint                                 |
| `manifest.txt`                  | sha256 of every artifact                             |

### Requirements

- Python 3.12. 3.14 does not work with torch/nemo.
- `uv` to build the venv.
- ~20 GB disk. Checkpoint 2.5 GB plus fp32 and int8 assets.
- Docker, to build the model images.

### Setup, once

```sh
make model-export-asr-venv
```

This builds `export/venv` with Python 3.12 and torch, plus the dependencies
from `export/asr/requirements.txt` (nemo_toolkit).

### Regenerate the assets

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

### Local attention: the patch

NeMo 3.0 exports `RelPositionMultiHeadAttentionLongformer` wrong. Its
sliding-chunks path rebuilds the attention band with `new_empty()` plus
in-place slice assignments. The torchscript exporter mis-compiles those and
the encoder outputs garbage.

`export.py` replaces that forward with a full-attention + band-mask version.
Numerically equivalent (max diff ~1e-3) and uses only export-safe ops.

Caveat: the replacement is O(T²) in memory, not O(T·w). Fine for audio up to
tens of minutes. For hours, fix the sliding-chunks export instead.

### nemo128.onnx

Test reference only. The server never loads it; `internal/mel` handles the mel
frontend. torch>=2.x makes `torch.stft` return complex values, which
torchscript cannot trace, so the export is best-effort and usually skipped.
To regenerate it, patch the preprocessor stft to `return_complex=False`.

## Diarization

One streaming step per chunk. These are the tensors the Go server feeds
(`internal/diarization/model.go`); the AOSC/FIFO speaker cache is not in the
graph, it lives in Go.

| tensor            | shape                      | meaning                                       |
| ----------------- | -------------------------- | --------------------------------------------- |
| `input_features`  | `(1, T_mel, 128)`          | log-mel features of the chunk                 |
| `cached_embeds`   | `(1, C, 512)`              | speaker-cache embeddings, empty on chunk 0    |
| `attention_mask`  | `(1, C + T_mel/8)`         | valid encoder frames                          |
| `logits`          | `(1, C*8 + T_mel, 8)`      | speaker activity logits at the mel frame rate |
| `chunk_embeds`    | `(1, T_mel/8, 512)`        | embeddings to push into the speaker cache     |
| `silence_embeds`  | `(512,)`                   | learned silence embedding                     |

### What the export produces

| file                             | what                                           |
| -------------------------------- | ---------------------------------------------- |
| `config.json`                    | model config, from the checkpoint repo         |
| `processor_config.json`          | processor config, from the checkpoint repo     |
| `preprocessor_config.json`       | feature extractor config, generated here       |
| `model.onnx`                     | fp32 step graph (~380 MB)                      |
| `model_quantized.onnx` + `_data` | int8 step graph (~110 MB)                      |
| `manifest.txt`                   | revision, hashes, quantization scheme          |

`preprocessor_config.json` is not published upstream. The export builds it from
the feature extractor defaults plus `audio_config.num_mel_bins`.

### Regenerate

```sh
make model-export-diarization-venv
make model-export-diarization
```

CPU only, about a minute, ~6 GB of disk. The checkpoint lands in
`export/diarization/checkpoint/`, pinned by commit and by content hash
(`MODEL_REVISION`, `CHECKPOINT_SHA256`). Upstream has no tags, only a moving
`main`.

### Quantization

8-bit weight-only (`MatMulNBits`, block 32, symmetric), same scheme as the
conversion we used to ship. `/embedder/projection/MatMul` stays fp32, as
upstream does, because its output feeds the next chunk's cache.

Running the export twice gives the same sha256.

### Tests

`export.py` fails unless both graphs match torch: fp32 cosine >= 0.99999, int8
>= 0.999, four shapes including an empty cache.

Parity against the old conversion, opt-in:

```sh
DIAR_REF_DIR=<old> DIAR_NEW_DIR=<new> go test ./internal/diarization/ -run TestParity -v
```

It uses the 2-speaker sample plus 5 minutes of the long reference clip. The old
conversion is not in `models/`; fetch a copy when needed:

```sh
REF=onnx-community/Nemotron-3-Diarization-ONNX
REV=353b6f8ad2cac3580e982d7fbdf0a010786b0406
REF_DIR=/tmp/diar-reference-int8 && mkdir -p $REF_DIR
for f in config.json preprocessor_config.json processor_config.json; do
  curl -sL -o $REF_DIR/$f "https://huggingface.co/$REF/resolve/$REV/$f"
done
for f in model_quantized.onnx model_quantized.onnx_data; do
  curl -sL -o $REF_DIR/$f "https://huggingface.co/$REF/resolve/$REV/onnx/$f"
done
```

Measured on this export:

| comparison | result |
| ---------- | ------ |
| fp32 vs reference fp32 | bit-identical |
| int8 vs reference int8, raw step | cosine 0.99996, decisions agree on 99.996% of frames |
| int8, 5 min | dominant speaker agrees on 100% of frames |
| int8, turns | same 66 segments, same speakers, within one frame |

Single model, no reference needed:

```sh
DIAR_DIR=$PWD/models/diarization/int8 go test ./internal/diarization/ -run TestDiarizeSampleAudio -v
```

Expected on the bundled audio: four turns, `spk=0` (0.6-14.2 s), `spk=1`
(15.7-29.1 s), `spk=0` (30.5-44.1 s), `spk=1` (45.6-59.0 s).

## Model images

The assets are shipped as OCI images on GitHub Container Registry, one image
per family and precision:

| image                                            | contents                          |
| ------------------------------------------------ | --------------------------------- |
| `ghcr.io/achetronic/parakeet-models-asr-int8`           | int8 encoder + decoder + vocab    |
| `ghcr.io/achetronic/parakeet-models-asr-fp32`           | fp32 encoder + decoder + vocab    |
| `ghcr.io/achetronic/parakeet-models-diarization-int8`   | int8 step graph + configs         |
| `ghcr.io/achetronic/parakeet-models-diarization-fp32`   | fp32 step graph + configs         |

Build locally without pushing:

```sh
make model-build-oci-images
```

Build and push:

```sh
docker login ghcr.io -u <user> --password-stdin   # once, with a PAT
make MODEL_TAG=v1.0.0 model-push-oci-images
```

`MODEL_TAG` defaults to `latest`. The images must exist before
`make model-download` or the Dockerfiles can pull them.

Both Dockerfiles take `MODEL_VERSION` (default `latest`) and the release
workflow does not override it, so a release build consumes whatever is
published as `:latest`. Push the model images before building the app images
for a new version, or pass `MODEL_VERSION=<tag>` explicitly.

## CI regeneration

`.github/workflows/export-models.yaml` regenerates every family and pushes all
four images. Trigger it manually with a tag. Empty tag means checkout `master`
and publish `:latest`; a tag checks out that ref and publishes under that tag.
