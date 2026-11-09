<div align="center">
  <img src="docs/img/parakeet.png" width="120">
  <h1>Parakeet</h1>
  <p>A high-performance automatic speech recognition (ASR) server written in Go.<br>
  It uses NVIDIA's Parakeet TDT 0.6B model through ONNX Runtime to provide speech-to-text transcription via an OpenAI Whisper-compatible API.</p>
</div>

<br>

## Table of Contents

- [Overview](#overview)
- [Model Architecture](#model-architecture)
- [Parakeet vs Whisper](#parakeet-vs-whisper)
- [Requirements](#requirements)
  - [Installing ONNX Runtime](#installing-onnx-runtime)
- [Installation](#installation)
  - [From Release Binary](#from-release-binary)
  - [From Source](#from-source)
  - [Using Docker](#using-docker)
  - [GPU Inference (CUDA)](#gpu-inference-cuda)
- [Configuration](#configuration)
  - [Command Line Flags](#command-line-flags)
  - [Environment Variables](#environment-variables)
  - [Long Audio](#long-audio)
  - [Speaker Diarization](#speaker-diarization)
  - [Model Files](#model-files)
- [API Reference](#api-reference)
  - [Authentication](#authentication)
  - [Transcribe Audio](#transcribe-audio)
  - [Segments and timing](#segments-and-timing)
  - [Speaker Diarization](#speaker-diarization-1)
  - [Streaming](#streaming)
  - [List Models](#list-models)
  - [Health Check](#health-check)
- [Development](#development)
- [Troubleshooting](#troubleshooting)
- [License](#license)
- [Credits](#credits)

## Overview

Parakeet ASR Server provides a lightweight, production-ready speech recognition service without Python dependencies. It exposes an API compatible with OpenAI's Whisper, making it a drop-in replacement for applications already using that interface.

Key features:

- OpenAI Whisper-compatible REST API
- API key authentication (optional, via environment variable)
- Streaming transcriptions via Server-Sent Events (OpenAI-compatible `transcript.text.delta` / `transcript.text.done`)
- ONNX Runtime inference (CPU and optional CUDA GPU)
- No Python dependency at runtime (ffmpeg is an optional system dependency for non-WAV audio)
- Structured logging with `slog` (text and JSON formats, configurable log level)
- Support for multiple response formats (JSON, text, SRT, VTT)
- Timed segments in `verbose_json`, and one SRT/VTT cue per segment, derived from the decoder's own token positions
- Multilingual support (English and 25+ languages)
- Quantized model support for reduced memory footprint
- Automatic audio conversion for non-WAV formats (MP3, OGG, WebM, FLAC, M4A, AAC, Opus, ...) when ffmpeg is installed
- Speaker diarization (opt-in): attribute each word to a speaker and return speaker-tagged turns

## Model Architecture

This server uses the [NVIDIA Parakeet TDT 0.6B](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) model, exported to ONNX from the official NVIDIA checkpoint.

The architecture consists of:

- **Encoder**: Conformer-based encoder with 1024-dimensional output. Processes 128-dimensional mel filterbank features with 8x temporal subsampling.
- **Decoder**: Token-and-Duration Transducer (TDT) decoder that jointly predicts tokens and their durations. Uses a 2-layer LSTM with 640-dimensional hidden state.
- **Vocabulary**: 8193 SentencePiece tokens including a blank token for CTC-style decoding.

The int8 quantized models require approximately 670MB of disk space and 2GB of RAM during inference.

## Parakeet vs Whisper

| Aspect        | Parakeet TDT                    | OpenAI Whisper                        |
| ------------- | ------------------------------- | ------------------------------------- |
| Architecture  | Conformer encoder + TDT decoder | Transformer encoder-decoder           |
| Decoding      | Non-autoregressive (parallel)   | Autoregressive (sequential)           |
| Speed         | Faster inference due to TDT     | Slower due to autoregressive decoding |
| Model size    | 0.6B parameters                 | 0.04B - 1.5B parameters               |
| Training data | NeMo ASR datasets               | 680K hours web audio                  |
| Primary focus | Accuracy and speed balance      | Multilingual robustness               |
| Timestamps    | Duration-based prediction       | Attention-based alignment             |

Parakeet TDT uses Token-and-Duration Transducer decoding, which predicts both the token and how many encoder frames to advance in a single step. This allows for faster inference compared to traditional autoregressive decoders while maintaining competitive accuracy.

## Requirements

- **ONNX Runtime 1.25.x or later** (required at runtime)
- Parakeet TDT ONNX models (downloaded separately)
- **ffmpeg** (optional) — enables automatic conversion of MP3, OGG, WebM, FLAC, M4A, AAC, Opus and any other ffmpeg-supported format. When ffmpeg is not present, only WAV input is accepted and non-WAV uploads return a 400 error. The official Docker image already ships with ffmpeg.

For building from source:

- Go 1.25 or later

### Installing ONNX Runtime

ONNX Runtime is required to run the inference. Choose the installation method for your Linux distribution:

#### Ubuntu / Debian

```bash
# Option 1: Download from GitHub releases (recommended)
curl -L -o onnxruntime.tgz "https://github.com/microsoft/onnxruntime/releases/download/v1.25.1/onnxruntime-linux-x64-1.25.1.tgz"
tar xzf onnxruntime.tgz
sudo cp -a onnxruntime-linux-x64-1.25.1/lib/* /usr/local/lib/
sudo ldconfig

# Option 2: Using apt (if available in your version)
sudo apt update
sudo apt install libonnxruntime-dev
```

#### Fedora / RHEL / CentOS

```bash
# Download from GitHub releases
curl -L -o onnxruntime.tgz "https://github.com/microsoft/onnxruntime/releases/download/v1.25.1/onnxruntime-linux-x64-1.25.1.tgz"
tar xzf onnxruntime.tgz
sudo cp -a onnxruntime-linux-x64-1.25.1/lib/* /usr/local/lib/
sudo ldconfig
```

#### Arch Linux

```bash
# From AUR
yay -S onnxruntime

# Or manually
curl -L -o onnxruntime.tgz "https://github.com/microsoft/onnxruntime/releases/download/v1.25.1/onnxruntime-linux-x64-1.25.1.tgz"
tar xzf onnxruntime.tgz
sudo cp -a onnxruntime-linux-x64-1.25.1/lib/* /usr/local/lib/
sudo ldconfig
```

#### Alpine Linux

```bash
apk add onnxruntime
```

#### Manual Installation (any distro)

```bash
# Download and extract
curl -L -o onnxruntime.tgz "https://github.com/microsoft/onnxruntime/releases/download/v1.25.1/onnxruntime-linux-x64-1.25.1.tgz"
tar xzf onnxruntime.tgz

# Install to /usr/local
sudo cp -a onnxruntime-linux-x64-1.25.1/lib/* /usr/local/lib/
sudo cp -r onnxruntime-linux-x64-1.25.1/include/* /usr/local/include/
sudo ldconfig

# Or set environment variable to use from current directory
export ONNXRUNTIME_LIB=$(pwd)/onnxruntime-linux-x64-1.25.1/lib/libonnxruntime.so
```

#### ARM64 (Raspberry Pi, etc.)

```bash
curl -L -o onnxruntime.tgz "https://github.com/microsoft/onnxruntime/releases/download/v1.25.1/onnxruntime-linux-aarch64-1.25.1.tgz"
tar xzf onnxruntime.tgz
sudo cp -a onnxruntime-linux-aarch64-1.25.1/lib/* /usr/local/lib/
sudo ldconfig
```

#### Verify Installation

```bash
# Check if library is found
ldconfig -p | grep onnxruntime

# Or find the library manually
find /usr -name "libonnxruntime.so*" 2>/dev/null
```

If the library is installed in a non-standard location, set the `ONNXRUNTIME_LIB` environment variable:

```bash
export ONNXRUNTIME_LIB=/path/to/libonnxruntime.so
```

## Installation

### From Release Binary

Clone the repository and download the release binary for your platform. The
clone provides the `Makefile` that downloads the models.

```bash
# Clone the repository (provides the Makefile used to download the models)
git clone https://github.com/achetronic/parakeet.git
cd parakeet

# Download the release binary for your platform
curl -L -o parakeet https://github.com/achetronic/parakeet/releases/latest/download/parakeet-linux-amd64
chmod +x parakeet

# Download models
make model-download            # ASR + VAD + diarization, int8 + fp32
# Or just the ASR int8 models:
make model-download-asr-int8   # int8 quantized models (recommended, ~670MB)

# Run (requires ONNX Runtime - see Installing ONNX Runtime section above)
./parakeet -port 5092 -models ./models
```

### From Source

```bash
# Clone the repository
git clone https://github.com/achetronic/parakeet.git
cd parakeet

# Download models
make model-download            # everything (ASR + VAD + diarization, both precisions)
# Or just ASR int8:
make model-download-asr-int8   # int8 quantized models (recommended)

# Build
make build

# Run
./bin/parakeet
```

### Using Docker

The Docker image that is published in the repository is ready to use. It includes ONNX Runtime and models.

```bash

# Run the container
docker run -d \
  --name parakeet \
  -p 5092:5092 \
  ghcr.io/achetronic/parakeet:latest

# Speaker diarization is bundled in the image but disabled by default;
# enable it per deployment:
docker run -d \
  --name parakeet \
  -p 5092:5092 \
  -e PARAKEET_ENABLE_DIARIZATION=true \
  ghcr.io/achetronic/parakeet:latest
```

#### Docker Compose

```yaml
services:
  parakeet:
    image: ghcr.io/achetronic/parakeet:latest
    ports:
      - "5092:5092"
    environment:
      - PARAKEET_API_KEY=your-secret-key # optional
      - PARAKEET_WORKERS=2
    restart: unless-stopped
```

### GPU Inference (CUDA)

Parakeet can offload inference to an NVIDIA GPU through the ONNX Runtime CUDA
execution provider. GPU support ships as a dedicated image with the `-cuda`
tag suffix; it bundles the GPU build of ONNX Runtime and the fp32 models on a
CUDA base image.

**Prerequisites:** an NVIDIA GPU with up-to-date drivers and the
[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
installed on the host. The CUDA image is `linux/amd64` only.

```bash
docker run -d \
  --name parakeet \
  --gpus all \
  -p 5092:5092 \
  ghcr.io/achetronic/parakeet:latest-cuda
```

The CUDA image runs on the GPU out of the box. To pick which GPUs (or MIG
slices) to use, list their ids; workers spread across them:

```bash
./parakeet -gpu cuda -gpu-devices 0,1,2
```

Same via env: `PARAKEET_GPU=cuda` and `PARAKEET_GPU_DEVICES=0,1,2`.

If the GPU can't be used (missing CUDA, wrong driver), the server refuses to
start instead of silently falling back to CPU.

> [!NOTE]
> fp32 is the GPU-appropriate precision and is the default for the CUDA image.
> CPU images and CPU mode are unaffected by these flags.

**Memory.** Two things decide how much VRAM you use:

- **Audio length.** Files of an hour or more can exceed the card and fail to
  allocate. Chunk them (ffmpeg `-f segment`) or use the CPU image.
- **`-workers`.** Each concurrent worker keeps its own model in VRAM: ~670 MB
  (int8) or ~2.5 GB (fp32). Set it to what your card, or your `-gpu-devices`
  slices, actually hold.

## Configuration

### Command Line Flags

| Flag                          | Description                                                              | Default                    | Example                                |
| ----------------------------- | ------------------------------------------------------------------------ | -------------------------- | -------------------------------------- |
| `-port`                       | HTTP server port                                                         | `5092`                     | `-port 8080`                           |
| `-models`                     | Path to models directory (organized by family)                          | `./models`                 | `-models /opt/parakeet/models`         |
| `-model-precision`            | Model precision: `int8` or `fp32`                                       | `int8`                     | `-model-precision fp32`               |
| `-log-level`                  | Log level: debug, info, warn, error                                      | `info`                     | `-log-level debug`                     |
| `-log-format`                 | Log output format: text or json                                          | `text`                     | `-log-format json`                     |
| `-workers`                    | Concurrent transcriptions (each ~2GB RAM for int8, ~6GB for fp32)        | `4`                        | `-workers 2`                           |
| `-ffmpeg`                     | Enable ffmpeg fallback for non-WAV audio                                 | `true`                     | `-ffmpeg=false`                        |
| `-ffmpeg-path`                | Path to the ffmpeg binary (empty = resolve from `PATH`)                  | ``                         | `-ffmpeg-path /usr/bin/ffmpeg`         |
| `-ffmpeg-timeout`             | Maximum wall-clock time for a single ffmpeg conversion                   | `60s`                      | `-ffmpeg-timeout 30s`                  |
| `-gpu`                        | Execution provider: `cpu` or `cuda`                                      | `cpu`                      | `-gpu cuda`                            |
| `-gpu-devices`                | Comma-separated CUDA device ids to spread workers over                   | `0`                        | `-gpu-devices 0,1,2`                   |
| `-long-audio`                 | Split audio over the model limit into chunks instead of rejecting it     | `false`                    | `-long-audio`                          |
| `-chunk-seconds`              | Sliding-window size for long audio, in seconds                           | `60`                       | `-chunk-seconds 120`                   |
| `-chunk-overlap-seconds`      | Overlap between consecutive chunks, in seconds                           | `15`                       | `-chunk-overlap-seconds 20`            |
| `-disable-vad-based-chunking` | Disable the Silero VAD chunk-boundary layer (falls back to mel energy)   | `false`                    | `-disable-vad-based-chunking`          |
| `-disable-mel-based-chunking` | Disable the mel-energy chunk-boundary layer (falls back to the midpoint) | `false`                    | `-disable-mel-based-chunking`          |
| `-vad-model-path`             | Path to the Silero VAD ONNX model                                        | `<models>/vad/silero_vad.onnx` | `-vad-model-path /opt/silero_vad.onnx` |
| `-enable-diarization`         | Load the diarization model and allow `diarize=true`                      | `false`                    | `-enable-diarization`                  |

**Examples:**

```bash
# Basic usage
./parakeet

# Custom port and models directory
./parakeet -port 8080 -models /opt/models

# Enable debug logging for troubleshooting
./parakeet -log-level debug

# JSON logs for production (e.g. log aggregation with ELK, Loki, CloudWatch)
./parakeet -log-format json

# JSON logs with debug level
./parakeet -log-format json -log-level debug

# Suppress ONNX Runtime schema warnings (stderr) while keeping debug logs
./parakeet -log-level debug 2>&1 | grep -v "Schema error"
```

### Environment Variables

Every command-line flag also reads from an environment variable: take the flag
name, uppercase it and replace dashes with underscores, then prefix it with
`PARAKEET_`. So `-log-level` maps to `PARAKEET_LOG_LEVEL`, `-ffmpeg-timeout` to
`PARAKEET_FFMPEG_TIMEOUT`, and so on. An explicit flag always overrides its env
var (precedence: **CLI flag > env var > default**); an invalid env value is
ignored with a warning and the default is kept.

The Docker images bake their operational defaults as `PARAKEET_*` env vars, so
they survive when you pass your own flags.

A few variables have no flag equivalent:

| Variable           | Description                                 | Default               |
| ------------------ | ------------------------------------------- | --------------------- |
| `ONNXRUNTIME_LIB`  | Path to libonnxruntime.so                   | Auto-detected         |
| `PARAKEET_API_KEY` | API key for `/v1/*` endpoint authentication | Empty (auth disabled) |

### Long Audio

The model's encoder tops out at 400 seconds of audio in a single pass. By
default, longer input is rejected with a clear error (and a log line pointing
here). Pass `-long-audio` to split it into overlapping windows
(`-chunk-seconds`, `-chunk-overlap-seconds`), transcribe each, and stitch the
results, dropping the overlap so words at the seams are not duplicated. Files
under the chunk size are transcribed in one pass either way.

**How chunk boundaries are chosen.** A blind split in the middle of an overlap
can fall mid-word and make that word show up twice or vanish at the seam. To
avoid this, the overlap is split on silence using a cascade (each layer falls
through to the next when it cannot decide):

1. **Silero VAD** picks the centre of the longest silence in the overlap. This
   needs `silero_vad.onnx` in the models directory (downloaded by `make model-download`;
   see below). A missing model is not fatal: it warns once and falls back to the
   next layer.
2. **Mel energy** picks the quietest point of the already-extracted features.
3. **Midpoint** is the final fallback (the original behaviour).

A second, always-on safety net removes any duplicate or colliding tokens right
at each seam. You can turn off individual layers with
`-disable-vad-based-chunking` / `-disable-mel-based-chunking`.

### Speaker Diarization

The server can optionally distinguish speakers using [NVIDIA Nemotron-3-Diarization](https://huggingface.co/nvidia/Nemotron-3-Diarization) (100M parameters, up to 8 speakers), running as a second ONNX model. Diarization is **disabled by default**; enable it with `-enable-diarization` (or `PARAKEET_ENABLE_DIARIZATION=true`) and request it per call with `diarize=true`. The model is bundled by `make model-download` and in the Docker images.

### Model Files

Models are organized by family, precision inside. The server reads the family
from `-models` (default `./models`) and the precision from `-model-precision`
(default `int8`).

```
models/
├── asr/int8/     config.json, vocab.txt, encoder-full.int8.onnx, encoder-local.int8.onnx, decoder_joint-model.int8.onnx
├── asr/fp32/     config.json, vocab.txt, encoder-full.onnx + .data, encoder-local.onnx + .data, decoder_joint-model.onnx
├── vad/          silero_vad.onnx
└── diarization/int8/ + diarization/fp32/   Nemotron-3-Diarization
```

The server uses the **local-attention** encoder (`encoder-local.*`). The
full-attention encoder (`encoder-full.*`) ships in the distribution image but
is not loaded at runtime.

| File (in `asr/<precision>/`)        | Size   | Description                    |
| ----------------------------------- | ------ | ------------------------------ |
| `config.json`                       | 97 B   | Model configuration            |
| `vocab.txt`                         | 94 KB  | SentencePiece vocabulary       |
| `encoder-local.int8.onnx`           | 614 MB | int8 encoder (local attention) |
| `encoder-full.int8.onnx`            | 652 MB | int8 encoder (full attention)  |
| `decoder_joint-model.int8.onnx`     | 54 MB  | int8 TDT decoder               |
| `encoder-local.onnx` + `.data`      | 2.4 GB | fp32 encoder (local attention) |
| `encoder-full.onnx` + `.data`       | 2.4 GB | fp32 encoder (full attention)  |
| `decoder_joint-model.onnx`          | 72 MB  | fp32 TDT decoder               |

`vad/silero_vad.onnx` ([snakers4/silero-vad](https://github.com/snakers4/silero-vad), MIT, pinned to v6.2.1) is precision-independent. Used only for chunk boundaries in long-audio mode; missing is a graceful degrade to mel energy.

Speaker diarization (optional, only when `-enable-diarization` is set) needs the
Nemotron-3-Diarization model under `diarization/<precision>/`:

| File (in `diarization/<precision>/`)   | Size   | Description                      |
| -------------------------------------- | ------ | -------------------------------- |
| `model_quantized.onnx` + `.onnx_data`  | 120 MB | int8 diarization encoder         |
| or `model.onnx` + `model.onnx_data`    | 400 MB | fp32 diarization encoder         |
| `config.json`                          | 1.6 KB | Streaming / speaker-cache config |
| `preprocessor_config.json`             | 339 B  | Feature extractor config         |
| `processor_config.json`                | 623 B  | Processor config                 |

Everything is downloaded by `make model-download`.

## API Reference

### Authentication

When `PARAKEET_API_KEY` is set, all `/v1/*` endpoints require an `Authorization` header:

```bash
curl -H "Authorization: Bearer YOUR_API_KEY" http://localhost:5092/v1/models
```

The `/health` endpoint is always unauthenticated.

### Transcribe Audio

```
POST /v1/audio/transcriptions
```

Transcribes audio into text. Compatible with OpenAI's Whisper API.

**Request**

Content-Type: `multipart/form-data`

| Parameter         | Type   | Required | Description                                                                                  |
| ----------------- | ------ | -------- | -------------------------------------------------------------------------------------------- |
| `file`            | file   | Yes      | Audio file (WAV always supported; MP3/OGG/WebM/FLAC/M4A/AAC/Opus via ffmpeg, max 25MB)       |
| `model`           | string | No       | Model name (accepted but ignored)                                                            |
| `language`        | string | No       | ISO-639-1 language code (default: en)                                                        |
| `response_format` | string | No       | Output format: json, text, srt, vtt, verbose_json (the last three carry per-segment timings) |
| `stream`          | bool   | No       | When `true`, stream the transcription as Server-Sent Events (see Streaming below)            |
| `prompt`          | string | No       | Accepted but ignored                                                                         |
| `temperature`     | float  | No       | Accepted but ignored                                                                         |
| `diarize`         | bool   | No       | When `true`, return speaker-attributed turns (requires `-enable-diarization`)                |

**Response**

JSON format (default):

```json
{
  "text": "transcribed text here"
}
```

Verbose JSON format:

```json
{
  "task": "transcribe",
  "language": "en",
  "duration": 12.48,
  "text": "I suppose I should take you back, really. I was twelve years old.",
  "segments": [
    {
      "id": 0,
      "seek": 800,
      "start": 0.8,
      "end": 3.44,
      "text": "I suppose I should take you back, really.",
      "tokens": [412, 1093, 301, 288, 117, 2140, 7],
      "temperature": 0,
      "avg_logprob": -0.118,
      "compression_ratio": 1.94,
      "no_speech_prob": 0
    },
    {
      "id": 1,
      "seek": 4320,
      "start": 4.32,
      "end": 6.16,
      "text": "I was twelve years old.",
      "tokens": [301, 412, 1904, 604, 7],
      "temperature": 0,
      "avg_logprob": -0.071,
      "compression_ratio": 1.62,
      "no_speech_prob": 0
    }
  ]
}
```

#### Segments and timing

`verbose_json` reports real segment timings, and `srt` / `vtt` emit one cue per
segment rather than one block covering the whole file.

The timings come from the decoder itself. TDT tags every emitted token with the
encoder frame it was decoded at, plus the frame count its duration head
predicted for it. The server groups those tokens into segments, closing one on
sentence punctuation, on a silence gap of about 0.64 s, or at 220 tokens / 30 s,
and converts frames to seconds with the model's own rate (12.5 encoder frames
per second, 0.08 s each).

What each field means:

| Field               | Meaning                                                                      |
| ------------------- | ---------------------------------------------------------------------------- |
| `duration`          | Audio length measured from the decoded waveform, not from the uploaded bytes |
| `start` / `end`     | Segment bounds in seconds, `end` clamped to `duration`                       |
| `seek`              | Segment start in milliseconds, matching OpenAI's field                       |
| `tokens`            | SentencePiece token ids of the segment                                       |
| `avg_logprob`       | Mean log probability the decoder assigned to the segment's tokens            |
| `compression_ratio` | `len(text) / len(gzip(text))`, the same redundancy metric OpenAI reports     |
| `temperature`       | 0: decoding is greedy                                                        |
| `no_speech_prob`    | 0: the TDT decoder has no no-speech head                                     |

Segment texts concatenate back to the `text` field, so the two views of the same
transcription never disagree.

**Example**

```bash
curl -X POST http://localhost:5092/v1/audio/transcriptions \
  -H "Authorization: Bearer $PARAKEET_API_KEY" \
  -F file=@audio.wav \
  -F language=en \
  -F response_format=json
```

#### Speaker Diarization

When the server runs with `-enable-diarization`, pass `diarize=true` to get speaker-attributed turns. Each turn carries a `speaker` index (in arrival order), its `start`/`end` times, and its `text`.

```bash
curl -X POST http://localhost:5092/v1/audio/transcriptions \
  -H "Authorization: Bearer $PARAKEET_API_KEY" \
  -F file=@meeting.wav \
  -F diarize=true
```

```json
{
  "task": "transcribe",
  "language": "en",
  "duration": 59.2,
  "text": "In this video, I'm gonna teach you...",
  "segments": [
    {
      "speaker": 0,
      "start": 0.56,
      "end": 14.8,
      "text": "In this video, I'm gonna teach you..."
    },
    {
      "speaker": 1,
      "start": 15.68,
      "end": 29.68,
      "text": "Going along slushy country roads..."
    }
  ]
}
```

#### Streaming

Set `stream=true` to receive the transcription incrementally as
[Server-Sent Events](https://developer.mozilla.org/docs/Web/API/Server-sent_events),
following OpenAI's streaming transcription protocol. The audio is still
uploaded in full; the server emits each chunk of text as soon as it is
decoded, then a final event with the complete text.

Two event types are sent:

- `transcript.text.delta` — a piece of newly transcribed text.
- `transcript.text.done` — sent once at the end, with the full transcript.

**Example**

```bash
curl -N -X POST http://localhost:5092/v1/audio/transcriptions \
  -H "Authorization: Bearer $PARAKEET_API_KEY" \
  -F file=@audio.wav \
  -F stream=true
```

**Response** (`Content-Type: text/event-stream`):

```
event: transcript.text.delta
data: {"type":"transcript.text.delta","delta":" Ma"}

event: transcript.text.delta
data: {"type":"transcript.text.delta","delta":"ybe"}

event: transcript.text.done
data: {"type":"transcript.text.done","text":"Maybe next time, huh?"}
```

This is compatible with clients that speak OpenAI's streaming
transcription API, such as Wyoming OpenAI for Home Assistant.

### List Models

```
GET /v1/models
```

Returns available models. Returns `parakeet-tdt-0.6b` and `whisper-1` (alias for compatibility).

### Health Check

```
GET /health
```

Returns `{"status": "ok"}` if the server is running.

## Development

### Available Make Targets

Run `make help` for the full, up-to-date list of targets.

### Running Tests

```bash
make test

# With coverage
make test-coverage
open coverage.html
```

## Troubleshooting

### ONNX Runtime library not found

Install ONNX Runtime or set the library path:

```bash
export ONNXRUNTIME_LIB=/path/to/libonnxruntime.so
```

Common installation locations:

- `/usr/lib/libonnxruntime.so`
- `/usr/local/lib/libonnxruntime.so`
- `/opt/onnxruntime/lib/libonnxruntime.so`

### Encoder model not found

Download the models:

```bash
make model-download
```

### Out of memory errors

Use the int8 quantized models (default) instead of fp32. The int8 models require approximately 2GB of RAM versus 6GB for fp32.

### Unsupported audio format

WAV is always supported natively. Any other format (MP3, OGG, WebM, FLAC, M4A, AAC, Opus, ...) is transcoded on the fly to 16 kHz mono WAV using a local `ffmpeg` binary.

If the server responds with `400 Unsupported or malformed audio`:

1. Install `ffmpeg` and make sure it is in `PATH` (or pass `-ffmpeg-path /absolute/path/to/ffmpeg`). The official Docker image already includes ffmpeg.
2. Check the server logs. On startup you will see one of:
   - `ffmpeg conversion enabled binary=/usr/bin/ffmpeg timeout=60s` — ready.
   - `ffmpeg not found, non-WAV inputs will be rejected` — install it or disable conversion with `-ffmpeg=false` if you only need WAV.
3. As a manual alternative, convert client-side before uploading:

   ```bash
   ffmpeg -i input.mp3 -ar 16000 -ac 1 output.wav
   ```

Audio is detected by content (magic bytes), not by filename extension, so clients that upload files without an extension still work.

## License

- Code: MIT License
- Parakeet Model: [CC-BY-4.0](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3)

## Credits

- **NVIDIA** - Original [Parakeet TDT 0.6B](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) model
