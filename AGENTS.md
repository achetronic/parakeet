# AGENTS.md

This document helps AI agents work effectively in this codebase.

## Project Overview

**Parakeet ASR Server** - A Go-based automatic speech recognition (ASR) server using NVIDIA's Parakeet TDT 0.6B model in ONNX format. Provides an OpenAI Whisper-compatible API for audio transcription.

### Key Technologies
- **Language**: Go 1.25+
- **ML Runtime**: ONNX Runtime 1.21.x (CPU inference)
- **Model**: NVIDIA Parakeet TDT 0.6B (Conformer-based encoder with Token-and-Duration Transducer decoder)
- **API**: REST, OpenAI Whisper-compatible

## Essential Commands

```bash
# Build
make build                  # Build to ./bin/parakeet

# Run
make run                    # Build and run with debug mode
make run-dev                # Run with custom port (5092) for development
./bin/parakeet              # Run binary directly
./bin/parakeet -port 8080 -models /path/to/models -debug=true

# Download models
make models                 # Download int8 models (default, ~670MB)
make models-int8            # Download int8 quantized models
make models-fp32            # Download full precision models (~2.5GB)

# Test
make test                   # Run tests
make test-coverage          # Run with coverage

# Code quality
make fmt                    # Format code
make vet                    # Run go vet
make lint                   # Run all linters (vet + fmt)

# Dependencies
make deps                   # Download Go dependencies
make deps-tidy              # Tidy dependencies
make deps-onnxruntime       # Install ONNX Runtime library

# Docker
make docker-build-int8      # Build image with int8 models
make docker-build-fp32      # Build image with fp32 models
make docker-run-int8        # Run container with int8 models
make docker-run-fp32        # Run container with fp32 models

# Release
make release                # Build all platforms
make release-linux          # Build Linux binaries (amd64/arm64)
make release-darwin         # Build macOS binaries (amd64/arm64)
make release-windows        # Build Windows binary (amd64)
```

## Project Structure

```
parakeet/
├── main.go                 # Entry point, CLI flags, server initialization
├── internal/
│   ├── asr/
│   │   ├── transcriber.go  # ONNX inference pipeline, TDT decoding
│   │   ├── mel.go          # Mel filterbank feature extraction (FFT, windowing)
│   │   └── audio.go        # WAV parsing, resampling to 16kHz
│   └── server/
│       ├── server.go       # HTTP server, route setup, lifecycle management
│       ├── handlers.go     # API endpoint handlers, response formatting
│       └── types.go        # Request/response type definitions
├── models/                 # ONNX models (downloaded separately)
├── bin/                    # Build output directory
├── Makefile                # Build recipes
├── Dockerfile              # Multi-stage container build
├── .github/
│   └── workflows/
│       ├── ci.yaml         # CI pipeline (lint, test, build)
│       └── release.yaml    # Release pipeline (binaries, docker)
└── README.md
```

## Code Organization

### `main.go` (Entry Point)
- Parses CLI flags: `-port`, `-models`, `-debug`
- Creates and runs the server
- Default port: 5092, default models dir: `./models`

### `internal/server/` (HTTP Server Package)

#### `server.go`
- `Config` struct: Port, ModelsDir, Debug settings
- `Server` struct: wraps config, transcriber, and HTTP mux
- `New()` - Initializes transcriber and routes
- `Run()` - Starts HTTP listener
- `Close()` - Releases resources

#### `handlers.go`
- `handleTranscription()` - Main endpoint, parses multipart form, returns transcription
- `handleTranslation()` - Delegates to transcription (Parakeet is English-focused)
- `handleModels()` - Returns available models (parakeet-tdt-0.6b, whisper-1 alias)
- `handleHealth()` - Health check endpoint
- Response format helpers: `formatSRTTime()`, `formatVTTTime()`
- CORS and error response utilities

#### `types.go`
- `TranscriptionResponse` - Simple JSON response with text
- `VerboseTranscriptionResponse` - Detailed response with segments, timing
- `Segment` - Transcription segment with timing info
- `ErrorResponse`, `ErrorDetail` - OpenAI-compatible error format
- `ModelInfo`, `ModelsResponse` - Model listing types

### `internal/asr/` (ASR Package)

#### `transcriber.go`
- `DebugMode` - Global flag for verbose logging
- `Config` - Model configuration (features_size, subsampling_factor)
- `Transcriber` - Main inference struct
- `NewTranscriber()` - Loads config, vocab, initializes ONNX Runtime
- `Transcribe()` - Main entry: audio -> mel -> encoder -> TDT decode -> text
- `loadAudio()` - Format detection and parsing
- `runInference()` - Encoder ONNX session execution
- `tdtDecode()` - TDT greedy decoding loop with state management
- `tokensToText()` - Token IDs to text with cleanup

#### `mel.go`
- `MelFilterbank` - Mel-scale filterbank feature extractor
- `NewMelFilterbank()` - Creates filterbank with NeMo defaults (128 mels, 512 FFT)
- `Extract()` - Computes mel features with Hann windowing
- `normalize()` - Per-utterance mean/variance normalization
- `fft()` - Radix-2 Cooley-Tukey FFT implementation
- Mel/Hz conversion helpers

#### `audio.go`
- `parseWAV()` - WAV parser supporting multiple chunk layouts
- `convertToFloat32()` - Supports 8/16/24/32-bit PCM and 32-bit float
- `resample()` - Linear interpolation resampling to 16kHz

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/v1/audio/transcriptions` | Transcribe audio (OpenAI-compatible) |
| POST | `/v1/audio/translations` | Translate audio (delegates to transcription) |
| GET | `/v1/models` | List available models |
| GET | `/health` | Health check |

### Transcription Parameters
- `file` (required) - Audio file (multipart form, max 25MB)
- `model` - Accepted but ignored (only one model)
- `language` - ISO-639-1 code (default: "en")
- `response_format` - json, text, srt, vtt, verbose_json (default: "json")
- `prompt`, `temperature` - Accepted but ignored

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
▁token 123
```
- SentencePiece format with `▁` (U+2581) as word boundary marker
- Special token: `<blk>` (blank) at index 8192

## Important Gotchas

### ONNX Runtime Library
- Must be installed separately (not vendored)
- Set `ONNXRUNTIME_LIB` env var if not in standard paths
- Auto-detection checks common paths in Makefile and transcriber.go
- Use `make deps-onnxruntime` to install (requires sudo)
- Compatible version: 1.21.x for onnxruntime_go v1.19.0

### Model Files Required
- `encoder-model.int8.onnx` (~652MB) or `encoder-model.onnx` (~2.5GB)
- `decoder_joint-model.int8.onnx` (~18MB) or `decoder_joint-model.onnx` (~72MB)
- `config.json`, `vocab.txt`, `nemo128.onnx`
- Download via `make models` or manually from HuggingFace

### Audio Format Limitations
- Currently **only WAV** format is supported
- WebM, OGG, MP3, M4A return "requires ffmpeg conversion - not yet implemented"
- All audio resampled to 16kHz mono internally
- Minimum audio length: 100ms (1600 samples at 16kHz)

### Tensor Memory Management
- Tensors must be destroyed manually (no GC)
- The TDT decode loop creates/destroys tensors each iteration
- Memory usage: ~2GB RAM for int8 models, ~6GB for fp32

### API Compatibility Notes
- `model` parameter accepted but ignored (only one model)
- `prompt` and `temperature` parameters accepted but ignored
- `language` defaults to "en" if not specified
- Translation endpoint just calls transcription

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `ONNXRUNTIME_LIB` | Path to libonnxruntime.so | Auto-detect |

## Dependencies

From `go.mod`:
```
go 1.25.5
github.com/yalue/onnxruntime_go v1.19.0
```

No other external Go dependencies. Standard library used for HTTP, JSON, audio processing, FFT.

## CI/CD

### CI Pipeline (`.github/workflows/ci.yaml`)
- Runs on push/PR to main/master
- Jobs: lint (Go 1.22), test (Go 1.25), build (Go 1.25)
- Lint checks: go vet, gofmt

### Release Pipeline (`.github/workflows/release.yaml`)
- Triggers on version tags (v*)
- Builds binaries for linux/darwin/windows (amd64/arm64)
- Creates GitHub release with checksums
- Pushes Docker images to ghcr.io (int8 and fp32 variants)

### Docker Build
- Multi-stage build with golang:1.25-bookworm builder
- Runtime: debian:bookworm-slim with ONNX Runtime 1.21.0
- Models embedded in image during build
- Health check included
- Exposed port: 5092

## Common Tasks for Agents

### Adding a New Audio Format
1. Add case in `internal/asr/transcriber.go:loadAudio()`
2. Implement parser in `internal/asr/audio.go`
3. Ensure output is `[]float32` normalized to [-1, 1] at 16kHz

### Modifying API Response
1. Add/modify structs in `internal/server/types.go`
2. Update relevant handler in `internal/server/handlers.go`
3. Follow OpenAI response format conventions

### Adding a New Endpoint
1. Add handler method to `internal/server/handlers.go`
2. Register route in `internal/server/server.go:setupRoutes()`
3. Add types to `internal/server/types.go` if needed

### Changing Inference Parameters
- Encoder dim: `internal/asr/transcriber.go:247` (`encoderDim := int64(1024)`)
- LSTM state: `internal/asr/transcriber.go:314-315` (`stateDim`, `numLayers`)
- Max tokens per step: `internal/asr/transcriber.go:39` (`maxTokensPerStep: 10`)
- Mel features: `internal/asr/mel.go:25-27` (nFFT, hopLength, winLength)

### Adding a New Makefile Target
1. Add target with `## Description` comment for help
2. Use `@` prefix for silent commands
3. Add to `.PHONY` if not a file target

### Creating a Release
1. Tag with semver: `git tag v1.0.0`
2. Push tag: `git push origin v1.0.0`
3. Release pipeline builds and publishes automatically
