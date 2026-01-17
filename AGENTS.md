# AGENTS.md

This document helps AI agents work effectively in this codebase.

## Project Overview

**Parakeet ASR Server** - A Go-based automatic speech recognition (ASR) server using NVIDIA's Parakeet TDT 0.6B model in ONNX format. Provides an OpenAI Whisper-compatible API for audio transcription.

### Key Technologies
- **Language**: Go 1.21+
- **ML Runtime**: ONNX Runtime (CPU inference)
- **Model**: NVIDIA Parakeet TDT 0.6B (Conformer-based encoder with Token-and-Duration Transducer decoder)
- **API**: REST, OpenAI Whisper-compatible

## Essential Commands

```bash
# Build
make build

# Run
make run                    # Build and run
./parakeet-server           # Run binary directly
./parakeet-server -port 8080 -models /path/to/models

# Download models
make models                 # Download int8 models (default)
make models-fp32            # Download full precision models

# Test
make test                   # Run tests
make test-coverage          # Run with coverage

# Code quality
make fmt                    # Format code
make vet                    # Run go vet
make lint                   # Run all linters

# Docker
make docker-build           # Build image
make docker-run             # Run container

# Release
make release                # Build all platforms
```

## Project Structure

```
parakeet/
├── main.go                 # HTTP server, API handlers, response formatting
├── internal/
│   └── asr/
│       ├── transcriber.go  # ONNX inference pipeline, TDT decoding
│       ├── mel.go          # Mel filterbank feature extraction (FFT, windowing)
│       └── audio.go        # WAV parsing, resampling to 16kHz
├── models/                 # ONNX models (downloaded separately)
├── Makefile                # Build recipes
├── Dockerfile              # Container build
├── .github/
│   └── workflows/
│       ├── ci.yaml         # CI pipeline (lint, test, build)
│       └── release.yaml    # Release pipeline (binaries, docker)
└── README.md
```

## Code Organization

### `main.go` (HTTP Server)
- Handles OpenAI-compatible endpoints
- Routes: `/v1/audio/transcriptions`, `/v1/audio/translations`, `/v1/models`, `/health`
- Parses multipart form data (25MB max)
- Supports response formats: `json`, `text`, `srt`, `vtt`, `verbose_json`
- CORS enabled for all origins

### `internal/asr/transcriber.go` (Inference Pipeline)
- `NewTranscriber()` - Initializes ONNX Runtime, loads vocab, creates mel filterbank
- `Transcribe()` - Main entry point: audio -> mel features -> encoder -> TDT decoder -> text
- `tdtDecode()` - Token-and-Duration Transducer greedy decoding loop
- `tokensToText()` - Converts token IDs to text using vocab

### `internal/asr/mel.go` (Feature Extraction)
- Custom mel filterbank implementation (no external DSP library)
- FFT using Cooley-Tukey algorithm
- Hann windowing, log mel energies
- Per-utterance mean/variance normalization

### `internal/asr/audio.go` (Audio Processing)
- WAV parser supporting 8/16/24/32-bit PCM
- Stereo to mono conversion
- Linear interpolation resampling to 16kHz

## Code Patterns & Conventions

### Naming
- Go standard naming (camelCase for private, PascalCase for exported)
- Descriptive function names: `parseWAV`, `convertToFloat32`, `tdtDecode`
- Type suffixes for ONNX tensors: `inputTensor`, `outputTensor`, `lengthTensor`

### Error Handling
- Wrap errors with `fmt.Errorf("context: %w", err)`
- Return early on error
- Cleanup resources with `defer` (tensor.Destroy(), file.Close())

### ONNX Runtime Usage
- Create tensors with `ort.NewTensor(shape, data)`
- Use `ort.NewAdvancedSession()` for named inputs/outputs
- Always call `.Destroy()` on tensors and sessions after use
- Memory-conscious: tensors created and destroyed per inference step in decode loop

### Response Formats
- JSON structs use tags: `json:"field_name"` with `omitempty` where appropriate
- OpenAI-compatible response structures

## Model Architecture Details

### Encoder
- Conformer architecture with 1024-dim output
- Input: mel features [batch, 128 features, time]
- Output: encoded features [batch, 1024, time/8]
- Subsampling factor: 8x

### TDT Decoder
- Token-and-Duration Transducer
- Vocab size: 8193 tokens (8192 + blank)
- Duration classes: 5 (predicts how many encoder steps to advance)
- LSTM state: 2 layers x 640 dim
- Greedy decoding with max 10 tokens per timestep

### Vocab Format
```
_token 123
```
- SentencePiece format with `_` as word boundary marker
- Special token: `<blk>` (blank) at index 8192

## Important Gotchas

### ONNX Runtime Library
- Must be installed separately (not vendored)
- Set `ONNXRUNTIME_LIB` env var if not in standard paths
- Auto-detection checks: `/usr/lib/`, `/usr/local/lib/`, `/opt/onnxruntime/lib/`

### Model Files Required
- `encoder-model.int8.onnx` (652MB) or `encoder-model.onnx` (2.5GB)
- `decoder_joint-model.int8.onnx` (18MB) or `decoder_joint-model.onnx` (72MB)
- `config.json` and `vocab.txt`
- Download via `make models` or manually from HuggingFace

### Audio Format Limitations
- Currently **only WAV** format is supported
- WebM, OGG, MP3, M4A return "requires ffmpeg - not yet implemented"
- All audio resampled to 16kHz mono internally

### Tensor Memory Management
- Tensors must be destroyed manually (no GC)
- The TDT decode loop creates/destroys tensors each iteration
- Memory usage: ~2GB RAM for int8 models, ~6GB for fp32

### API Compatibility Notes
- `model` parameter accepted but ignored (only one model)
- `prompt` and `temperature` parameters accepted but ignored
- `language` defaults to "en" if not specified
- Translation endpoint (`/v1/audio/translations`) just calls transcription

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `ONNXRUNTIME_LIB` | Path to libonnxruntime.so | Auto-detect |

## Dependencies

From `go.mod`:
```
github.com/yalue/onnxruntime_go v1.13.0
```

No other external Go dependencies. Standard library used for HTTP, JSON, audio processing.

## CI/CD

### CI Pipeline (`.github/workflows/ci.yaml`)
- Runs on push/PR to main
- Jobs: lint (vet, fmt), test, build

### Release Pipeline (`.github/workflows/release.yaml`)
- Triggers on version tags (v*)
- Builds binaries for linux/darwin/windows (amd64/arm64)
- Creates GitHub release with checksums
- Pushes Docker image to ghcr.io

## Common Tasks for Agents

### Adding a New Audio Format
1. Add case in `internal/asr/transcriber.go:loadAudio()`
2. Implement parser in `internal/asr/audio.go`
3. Ensure output is `[]float32` normalized to [-1, 1] at 16kHz

### Modifying API Response
1. Add/modify structs in `main.go`
2. Update relevant handler function
3. Follow OpenAI response format conventions

### Changing Inference Parameters
- Encoder dim: `internal/asr/transcriber.go:223` (`encoderDim := int64(1024)`)
- LSTM state: `internal/asr/transcriber.go:282-283` (`stateDim`, `numLayers`)
- Max tokens per step: `internal/asr/transcriber.go:35` (`maxTokensPerStep`)

### Adding a New Makefile Target
1. Add target with `## Description` comment for help
2. Use `@` prefix for silent commands
3. Add to `.PHONY` if not a file target

### Creating a Release
1. Tag with semver: `git tag v1.0.0`
2. Push tag: `git push origin v1.0.0`
3. Release pipeline builds and publishes automatically
