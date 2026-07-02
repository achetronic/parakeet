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
- [Configuration](#configuration)
  - [Command Line Flags](#command-line-flags)
  - [Environment Variables](#environment-variables)
  - [Model Files](#model-files)
- [API Reference](#api-reference)
  - [Transcribe Audio](#transcribe-audio)
  - [Streaming](#streaming)
- [Development](#development)
- [Project Structure](#project-structure)
- [Troubleshooting](#troubleshooting)
- [License](#license)

## Overview

Parakeet ASR Server provides a lightweight, production-ready speech recognition service without Python dependencies. It exposes an API compatible with OpenAI's Whisper, making it a drop-in replacement for applications already using that interface.

Key features:

- OpenAI Whisper-compatible REST API
- API key authentication (optional, via environment variable)
- Streaming transcriptions via Server-Sent Events (OpenAI-compatible `transcript.text.delta` / `transcript.text.done`)
- ONNX Runtime inference (CPU)
- No Python dependency at runtime (ffmpeg is an optional system dependency for non-WAV audio)
- Structured logging with `slog` (text and JSON formats, configurable log level)
- Support for multiple response formats (JSON, text, SRT, VTT)
- Multilingual support (English and 25+ languages)
- Quantized model support for reduced memory footprint
- Automatic audio conversion for non-WAV formats (MP3, OGG, WebM, FLAC, M4A, AAC, Opus, ...) when ffmpeg is installed

## Model Architecture

This server uses the [NVIDIA Parakeet TDT 0.6B](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) model, converted to ONNX format by [istupakov](https://huggingface.co/istupakov/parakeet-tdt-0.6b-v3-onnx).

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
sudo cp onnxruntime-linux-x64-1.25.1/lib/* /usr/local/lib/
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
sudo cp onnxruntime-linux-x64-1.25.1/lib/* /usr/local/lib/
sudo ldconfig
```

#### Arch Linux

```bash
# From AUR
yay -S onnxruntime

# Or manually
curl -L -o onnxruntime.tgz "https://github.com/microsoft/onnxruntime/releases/download/v1.25.1/onnxruntime-linux-x64-1.25.1.tgz"
tar xzf onnxruntime.tgz
sudo cp onnxruntime-linux-x64-1.25.1/lib/* /usr/local/lib/
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
sudo cp -r onnxruntime-linux-x64-1.25.1/lib/* /usr/local/lib/
sudo cp -r onnxruntime-linux-x64-1.25.1/include/* /usr/local/include/
sudo ldconfig

# Or set environment variable to use from current directory
export ONNXRUNTIME_LIB=$(pwd)/onnxruntime-linux-x64-1.25.1/lib/libonnxruntime.so
```

#### ARM64 (Raspberry Pi, etc.)

```bash
curl -L -o onnxruntime.tgz "https://github.com/microsoft/onnxruntime/releases/download/v1.25.1/onnxruntime-linux-aarch64-1.25.1.tgz"
tar xzf onnxruntime.tgz
sudo cp onnxruntime-linux-aarch64-1.25.1/lib/* /usr/local/lib/
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

Download the latest release for your platform from the [Releases](https://github.com/achetronic/parakeet/releases) page.

```bash
# Linux (amd64)
curl -L -o parakeet https://github.com/achetronic/parakeet/releases/latest/download/parakeet-linux-amd64
chmod +x parakeet

# Download models (using Makefile)
make models           # int8 quantized models (recommended, ~670MB)
# Or for full precision:
make models-fp32      # fp32 models (~2.5GB)

# Run (requires ONNX Runtime - see Installing ONNX Runtime section above)
./parakeet -port 5092 -models ./models
```

### From Source

```bash
# Clone the repository
git clone https://github.com/achetronic/parakeet.git
cd parakeet

# Download models
make models           # int8 quantized models (recommended)
# Or: make models-fp32  # full precision models

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
```

#### Docker Compose

```yaml
version: "3.8"
services:
  parakeet:
    image: ghcr.io/achetronic/parakeet:latest
    ports:
      - "5092:5092"
    environment:
      - PARAKEET_API_KEY=your-secret-key # optional
      - PARAKEET_WORKERS=2
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:5092/health"]
      interval: 30s
      timeout: 3s
      retries: 3
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

The CUDA image enables the GPU by default (`-gpu cuda`). For a non-Docker
binary, or to select a specific device, use the `-gpu`/`-gpu-device` flags or
the `PARAKEET_GPU`/`PARAKEET_GPU_DEVICE` environment variables:

```bash
./parakeet -gpu cuda -gpu-device 0
# or, equivalently, via environment:
PARAKEET_GPU=cuda PARAKEET_GPU_DEVICE=0 ./parakeet
```

If the requested provider cannot be initialized (missing CUDA libraries, driver
or version mismatch), the server fails at startup rather than silently falling
back to CPU, so misconfiguration is visible immediately.

> [!NOTE]
> fp32 is the GPU-appropriate precision and is the default for the CUDA image.
> CPU images and CPU mode are unaffected by these flags.

**Memory and audio length.** The encoder processes the whole file in a single
pass, so peak memory scales with audio duration. On GPU this is bounded by VRAM:
very long inputs (roughly an hour or more on a 24 GB card) can exceed device
memory and fail with an ONNX Runtime allocation error. Segment long audio (for
example with `ffmpeg -i in.wav -f segment -segment_time 300 -ar 16000 -ac 1
chunk_%03d.wav`) or use a CPU image, which is bounded by system RAM instead.

## Configuration

### Command Line Flags

| Flag              | Description                                             | Default    | Example                        |
| ----------------- | ------------------------------------------------------- | ---------- | ------------------------------ |
| `-port`           | HTTP server port                                        | `5092`     | `-port 8080`                   |
| `-models`         | Path to models directory                                | `./models` | `-models /opt/parakeet/models` |
| `-log-level`      | Log level: debug, info, warn, error                     | `info`     | `-log-level debug`             |
| `-log-format`     | Log output format: text or json                         | `text`     | `-log-format json`             |
| `-workers`        | Concurrent inference workers (each ~670MB RAM for int8) | `4`        | `-workers 2`                   |
| `-ffmpeg`         | Enable ffmpeg fallback for non-WAV audio                | `true`     | `-ffmpeg=false`                |
| `-ffmpeg-path`    | Path to the ffmpeg binary (empty = resolve from `PATH`) | ``         | `-ffmpeg-path /usr/bin/ffmpeg` |
| `-ffmpeg-timeout` | Maximum wall-clock time for a single ffmpeg conversion  | `60s`      | `-ffmpeg-timeout 30s`          |
| `-gpu`            | Execution provider: `cpu` or `cuda`                     | `cpu`      | `-gpu cuda`                    |
| `-gpu-device`     | GPU device index for `cuda`                             | `0`        | `-gpu-device 1`                |
| `-long-audio`            | Split audio over the model limit into chunks instead of rejecting it | `false` | `-long-audio`         |
| `-chunk-seconds`         | Sliding-window size for long audio, in seconds    | `300`      | `-chunk-seconds 240`           |
| `-chunk-overlap-seconds` | Overlap between consecutive chunks, in seconds    | `15`       | `-chunk-overlap-seconds 10`    |
| `-disable-vad-based-chunking` | Disable the Silero VAD chunk-boundary layer (falls back to mel energy) | `false` | `-disable-vad-based-chunking` |
| `-disable-mel-based-chunking` | Disable the mel-energy chunk-boundary layer (falls back to the midpoint) | `false` | `-disable-mel-based-chunking` |
| `-vad-model-path`        | Path to the Silero VAD ONNX model                 | `<models>/silero_vad.onnx` | `-vad-model-path /opt/silero_vad.onnx` |

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
   needs `silero_vad.onnx` in the models directory (downloaded by `make models`;
   see below). A missing model is not fatal: it warns once and falls back to the
   next layer.
2. **Mel energy** picks the quietest point of the already-extracted features.
3. **Midpoint** is the final fallback (the original behaviour).

A second, always-on safety net removes any duplicate or colliding tokens right
at each seam. You can turn off individual layers with
`-disable-vad-based-chunking` / `-disable-mel-based-chunking`. See DD-014 in
`.agents/DESIGN_DECISIONS.md` for the full rationale.

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

### Model Files

The following files are required in the models directory:

| File                            | Size   | Description              |
| ------------------------------- | ------ | ------------------------ |
| `config.json`                   | 97 B   | Model configuration      |
| `vocab.txt`                     | 94 KB  | SentencePiece vocabulary |
| `nemo128.onnx`                  | 140 KB | Preprocessor graph       |
| `encoder-model.int8.onnx`       | 652 MB | Quantized encoder        |
| `decoder_joint-model.int8.onnx` | 18 MB  | Quantized TDT decoder    |
| `silero_vad.onnx`               | 2.3 MB | Silero VAD (chunk boundaries, long-audio only; optional) |

For full precision models, use `encoder-model.onnx` (requires `encoder-model.onnx.data`, 2.5GB total) and `decoder_joint-model.onnx` (72MB).

`silero_vad.onnx` ([snakers4/silero-vad](https://github.com/snakers4/silero-vad), MIT, pinned to release v6.2.1) is downloaded and checksum-verified by `make models`. It is only used to place chunk boundaries on silence in long-audio mode; if it is missing the server logs a warning once and falls back to mel-energy boundaries.

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

| Parameter         | Type   | Required | Description                                                                            |
| ----------------- | ------ | -------- | -------------------------------------------------------------------------------------- |
| `file`            | file   | Yes      | Audio file (WAV always supported; MP3/OGG/WebM/FLAC/M4A/AAC/Opus via ffmpeg, max 25MB) |
| `model`           | string | No       | Model name (accepted but ignored)                                                      |
| `language`        | string | No       | ISO-639-1 language code (default: en)                                                  |
| `response_format` | string | No       | Output format: json, text, srt, vtt, verbose_json                                      |
| `stream`          | bool   | No       | When `true`, stream the transcription as Server-Sent Events (see Streaming below)      |
| `prompt`          | string | No       | Accepted but ignored                                                                   |
| `temperature`     | float  | No       | Accepted but ignored                                                                   |

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
  "duration": 5.2,
  "text": "transcribed text here",
  "segments": [
    {
      "id": 0,
      "start": 0,
      "end": 5.2,
      "text": "transcribed text here"
    }
  ]
}
```

**Example**

```bash
curl -X POST http://localhost:5092/v1/audio/transcriptions \
  -H "Authorization: Bearer $PARAKEET_API_KEY" \
  -F file=@audio.wav \
  -F language=en \
  -F response_format=json
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

```bash
make help          # Show all available targets

# Build
make build         # Build the binary

# Development
make run           # Build and run
make run-dev       # Run with custom port (5092) for development
make clean         # Remove build artifacts

# Code quality
make fmt           # Format code
make vet           # Run go vet
make lint          # Run all linters (vet + fmt)
make test          # Run tests
make test-coverage # Run tests with coverage report

# Models
make models        # Download int8 models (default)
make models-int8   # Download int8 quantized models
make models-fp32   # Download full precision models

# Docker
make docker-build-int8  # Build Docker image with int8 models
make docker-build-fp32  # Build Docker image with fp32 models
make docker-build-cuda  # Build CUDA/GPU image with fp32 models
make docker-run-int8    # Run Docker container with int8 models
make docker-run-fp32    # Run Docker container with fp32 models
make docker-run-cuda    # Run CUDA/GPU container (needs --gpus all)

# Release
make release       # Build binaries for all platforms
```

### Running Tests

```bash
make test

# With coverage
make test-coverage
open coverage.html
```

## Project Structure

```
parakeet/
├── main.go                 # Entry point, CLI flags, logger setup
├── internal/
│   ├── asr/
│   │   ├── transcriber.go  # ONNX inference pipeline
│   │   ├── mel.go          # Mel filterbank feature extraction
│   │   └── audio.go        # WAV parsing and resampling
│   └── server/
│       ├── server.go       # HTTP server, auth middleware, lifecycle
│       ├── handlers.go     # API endpoint handlers
│       └── types.go        # Request/response type definitions
├── models/                 # ONNX models (not in repository)
├── .agents/                # AI agent documentation
├── Dockerfile
├── Makefile
├── .github/
│   └── workflows/
│       ├── ci.yaml         # CI pipeline
│       └── release.yaml    # Release pipeline
└── README.md
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
make models
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
- **Ivan Stupakov** ([@istupakov](https://github.com/istupakov)) - [ONNX conversion](https://huggingface.co/istupakov/parakeet-tdt-0.6b-v3-onnx) of the Parakeet model
