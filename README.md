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
- [Development](#development)
- [Project Structure](#project-structure)
- [Troubleshooting](#troubleshooting)
- [License](#license)

## Overview

Parakeet ASR Server provides a lightweight, production-ready speech recognition service without Python dependencies. It exposes an API compatible with OpenAI's Whisper, making it a drop-in replacement for applications already using that interface.

Key features:

- OpenAI Whisper-compatible REST API
- ONNX Runtime inference (CPU)
- No Python or external dependencies at runtime
- Support for multiple response formats (JSON, text, SRT, VTT)
- Multilingual support (English and 25+ languages)
- Quantized model support for reduced memory footprint

## Model Architecture

This server uses the [NVIDIA Parakeet TDT 0.6B](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) model, converted to ONNX format by [istupakov](https://huggingface.co/istupakov/parakeet-tdt-0.6b-v3-onnx).

The architecture consists of:

- **Encoder**: Conformer-based encoder with 1024-dimensional output. Processes 128-dimensional mel filterbank features with 8x temporal subsampling.
- **Decoder**: Token-and-Duration Transducer (TDT) decoder that jointly predicts tokens and their durations. Uses a 2-layer LSTM with 640-dimensional hidden state.
- **Vocabulary**: 8193 SentencePiece tokens including a blank token for CTC-style decoding.

The int8 quantized models require approximately 670MB of disk space and 2GB of RAM during inference.

## Parakeet vs Whisper

| Aspect        | Parakeet TDT                    | OpenAI Whisper                        |
|---------------|---------------------------------|---------------------------------------|
| Architecture  | Conformer encoder + TDT decoder | Transformer encoder-decoder           |
| Decoding      | Non-autoregressive (parallel)   | Autoregressive (sequential)           |
| Speed         | Faster inference due to TDT     | Slower due to autoregressive decoding |
| Model size    | 0.6B parameters                 | 0.04B - 1.5B parameters               |
| Training data | NeMo ASR datasets               | 680K hours web audio                  |
| Primary focus | Accuracy and speed balance      | Multilingual robustness               |
| Timestamps    | Duration-based prediction       | Attention-based alignment             |

Parakeet TDT uses Token-and-Duration Transducer decoding, which predicts both the token and how many encoder frames to advance in a single step. This allows for faster inference compared to traditional autoregressive decoders while maintaining competitive accuracy.

## Requirements

- **ONNX Runtime 1.17.0 or later** (required at runtime)
- Parakeet TDT ONNX models (downloaded separately)

For building from source:
- Go 1.21 or later

### Installing ONNX Runtime

ONNX Runtime is required to run the inference. Choose the installation method for your Linux distribution:

#### Ubuntu / Debian

```bash
# Option 1: Download from GitHub releases (recommended)
curl -L -o onnxruntime.tgz "https://github.com/microsoft/onnxruntime/releases/download/v1.17.0/onnxruntime-linux-x64-1.17.0.tgz"
tar xzf onnxruntime.tgz
sudo cp onnxruntime-linux-x64-1.17.0/lib/* /usr/local/lib/
sudo ldconfig

# Option 2: Using apt (if available in your version)
sudo apt update
sudo apt install libonnxruntime-dev
```

#### Fedora / RHEL / CentOS

```bash
# Download from GitHub releases
curl -L -o onnxruntime.tgz "https://github.com/microsoft/onnxruntime/releases/download/v1.17.0/onnxruntime-linux-x64-1.17.0.tgz"
tar xzf onnxruntime.tgz
sudo cp onnxruntime-linux-x64-1.17.0/lib/* /usr/local/lib64/
sudo ldconfig
```

#### Arch Linux

```bash
# From AUR
yay -S onnxruntime

# Or manually
curl -L -o onnxruntime.tgz "https://github.com/microsoft/onnxruntime/releases/download/v1.17.0/onnxruntime-linux-x64-1.17.0.tgz"
tar xzf onnxruntime.tgz
sudo cp onnxruntime-linux-x64-1.17.0/lib/* /usr/local/lib/
sudo ldconfig
```

#### Alpine Linux

```bash
apk add onnxruntime
```

#### Manual Installation (any distro)

```bash
# Download and extract
curl -L -o onnxruntime.tgz "https://github.com/microsoft/onnxruntime/releases/download/v1.17.0/onnxruntime-linux-x64-1.17.0.tgz"
tar xzf onnxruntime.tgz

# Install to /usr/local
sudo cp -r onnxruntime-linux-x64-1.17.0/lib/* /usr/local/lib/
sudo cp -r onnxruntime-linux-x64-1.17.0/include/* /usr/local/include/
sudo ldconfig

# Or set environment variable to use from current directory
export ONNXRUNTIME_LIB=$(pwd)/onnxruntime-linux-x64-1.17.0/lib/libonnxruntime.so
```

#### ARM64 (Raspberry Pi, etc.)

```bash
curl -L -o onnxruntime.tgz "https://github.com/microsoft/onnxruntime/releases/download/v1.17.0/onnxruntime-linux-aarch64-1.17.0.tgz"
tar xzf onnxruntime.tgz
sudo cp onnxruntime-linux-aarch64-1.17.0/lib/* /usr/local/lib/
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

The Docker image includes ONNX Runtime but requires models to be mounted at runtime.

```bash
# Pull the image
docker pull ghcr.io/achetronic/parakeet:latest

# Download models locally
mkdir -p models
make models

# Run the container
docker run -d \
  --name parakeet \
  -p 5092:5092 \
  -v $(pwd)/models:/models \
  ghcr.io/achetronic/parakeet:latest
```

Or build the image locally:

```bash
make docker-build
make docker-run
```

#### Docker Compose

```yaml
version: '3.8'
services:
  parakeet:
    image: ghcr.io/achetronic/parakeet:latest
    ports:
      - "5092:5092"
    volumes:
      - ./models:/models
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:5092/health"]
      interval: 30s
      timeout: 3s
      retries: 3
```

## Configuration

### Command Line Flags

| Flag | Description | Default | Example |
|------|-------------|---------|---------|
| `-port` | HTTP server port | `5092` | `-port 8080` |
| `-models` | Path to models directory | `./models` | `-models /opt/parakeet/models` |
| `-debug` | Enable debug logging (verbose output for troubleshooting) | `false` | `-debug` |

**Examples:**

```bash
# Basic usage
./parakeet

# Custom port and models directory
./parakeet -port 8080 -models /opt/models

# Enable debug logging for troubleshooting
./parakeet -debug

# Suppress ONNX Runtime schema warnings (stderr) while keeping debug logs
./parakeet -debug 2>&1 | grep -v "Schema error"
```

### Environment Variables

| Variable          | Description               | Default       |
|-------------------|---------------------------|---------------|
| `ONNXRUNTIME_LIB` | Path to libonnxruntime.so | Auto-detected |

### Model Files

The following files are required in the models directory:

| File                            | Size   | Description              |
|---------------------------------|--------|--------------------------|
| `config.json`                   | 97 B   | Model configuration      |
| `vocab.txt`                     | 94 KB  | SentencePiece vocabulary |
| `encoder-model.int8.onnx`       | 652 MB | Quantized encoder        |
| `decoder_joint-model.int8.onnx` | 18 MB  | Quantized TDT decoder    |

For full precision models, use `encoder-model.onnx` (requires `encoder-model.onnx.data`, 2.5GB total) and `decoder_joint-model.onnx` (72MB).

## API Reference

### Transcribe Audio

```
POST /v1/audio/transcriptions
```

Transcribes audio into text. Compatible with OpenAI's Whisper API.

**Request**

Content-Type: `multipart/form-data`

| Parameter         | Type   | Required | Description                                       |
|-------------------|--------|----------|---------------------------------------------------|
| `file`            | file   | Yes      | Audio file (WAV format, max 25MB)                 |
| `model`           | string | No       | Model name (accepted but ignored)                 |
| `language`        | string | No       | ISO-639-1 language code (default: en)             |
| `response_format` | string | No       | Output format: json, text, srt, vtt, verbose_json |
| `prompt`          | string | No       | Accepted but ignored                              |
| `temperature`     | float  | No       | Accepted but ignored                              |

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
  -F file=@audio.wav \
  -F language=en \
  -F response_format=json
```

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
make build-static  # Build statically linked binary

# Development
make run           # Build and run
make run-dev       # Run with development settings
make clean         # Remove build artifacts

# Code quality
make fmt           # Format code
make vet           # Run go vet
make lint          # Run all linters
make test          # Run tests
make test-coverage # Run tests with coverage report

# Models
make models        # Download int8 models (default)
make models-int8   # Download int8 quantized models
make models-fp32   # Download full precision models

# Docker
make docker-build  # Build Docker image
make docker-run    # Run Docker container

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
├── main.go                 # HTTP server and API handlers
├── internal/
│   └── asr/
│       ├── transcriber.go  # ONNX inference pipeline
│       ├── mel.go          # Mel filterbank feature extraction
│       └── audio.go        # WAV parsing and resampling
├── models/                 # ONNX models (not in repository)
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

Currently only WAV format is supported. Convert other formats using ffmpeg:

```bash
ffmpeg -i input.mp3 -ar 16000 -ac 1 output.wav
```

## License

- Code: MIT License
- Parakeet Model: [CC-BY-4.0](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3)

## Credits

- **NVIDIA** - Original [Parakeet TDT 0.6B](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) model
- **Ivan Stupakov** ([@istupakov](https://github.com/istupakov)) - [ONNX conversion](https://huggingface.co/istupakov/parakeet-tdt-0.6b-v3-onnx) of the Parakeet model
