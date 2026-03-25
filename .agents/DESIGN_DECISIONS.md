# Design Decisions

Architectural and design decisions made in this project.

## DD-001: OpenAI Whisper-Compatible API

**Context**: The server needs an HTTP API for audio transcription.

**Decision**: Implement an API compatible with OpenAI's Whisper endpoints (`/v1/audio/transcriptions`, `/v1/audio/translations`, `/v1/models`).

**Rationale**: Allows drop-in replacement for applications already using the OpenAI Whisper API. Reduces integration effort for adopters.

**Consequences**:
- `model`, `prompt`, and `temperature` parameters are accepted but ignored (single model, no prompt conditioning, deterministic greedy decoding)
- Translation endpoint delegates to transcription since Parakeet is English-focused
- Error responses follow OpenAI's format (`ErrorResponse`/`ErrorDetail` structs)

## DD-002: NVIDIA Parakeet TDT 0.6B Model

**Context**: Need a speech recognition model that balances accuracy and resource usage.

**Decision**: Use NVIDIA's Parakeet TDT 0.6B with Conformer encoder and Token-and-Duration Transducer (TDT) decoder.

**Rationale**: Compact model (~670MB int8) with strong English transcription accuracy. TDT decoder predicts both tokens and durations, enabling efficient greedy decoding without beam search.

**Architecture details**:
- Encoder: Conformer, 1024-dim output, 8x subsampling factor
- Decoder: TDT with 8193 vocab (8192 tokens + blank), 5 duration classes
- LSTM state: 2 layers × 640 dim
- Greedy decoding with max 10 tokens per timestep
- Vocab: SentencePiece format with `▁` (U+2581) word boundary, `<blk>` blank at index 8192

## DD-003: ONNX Runtime for Inference

**Context**: Need a runtime to execute the neural network model in Go.

**Decision**: Use ONNX Runtime via `onnxruntime_go` bindings (CPU only).

**Rationale**: ONNX is a portable, vendor-neutral format. CPU inference avoids GPU dependency, simplifying deployment. The Go bindings (`onnxruntime_go v1.19.0`) provide direct integration without CGo complexity.

**Consequences**:
- ONNX Runtime library (1.21.x) must be installed separately on the host
- `ONNXRUNTIME_LIB` env var needed if not in standard paths
- Tensors must be destroyed manually (no GC integration)
- Memory: ~2GB RAM for int8, ~6GB for fp32

## DD-004: WAV-Only Audio Input

**Context**: The server needs to accept audio files for transcription.

**Decision**: Support only WAV format natively. Other formats (WebM, OGG, MP3, M4A) return an error suggesting ffmpeg conversion.

**Rationale**: WAV parsing is straightforward with no external dependencies. Adding format support via ffmpeg would introduce a heavy system dependency. Keeps the binary self-contained.

**Consequences**:
- Clients must convert non-WAV audio before sending
- `loadAudio()` in `transcriber.go:207` returns explicit "not yet implemented" for unsupported formats
- Supports 8/16/24/32-bit PCM and 32-bit float WAV
- All audio resampled to 16kHz mono internally
- Minimum audio length: 100ms (1600 samples at 16kHz)

## DD-005: Pure Go Audio Processing

**Context**: Need mel-spectrogram feature extraction and audio resampling.

**Decision**: Implement FFT, mel filterbank, windowing, and resampling entirely in Go standard library. No external DSP dependencies.

**Rationale**: Zero external dependencies for audio processing. The NeMo-compatible defaults (128 mels, 512-point FFT, Hann window) ensure model compatibility.

**Consequences**:
- Radix-2 Cooley-Tukey FFT implementation in `mel.go`
- Linear interpolation resampling (simple but sufficient for speech)
- Per-utterance mean/variance normalization matches NeMo pipeline

## DD-006: Int8 Quantized Models as Default

**Context**: The model is available in both fp32 and int8 quantized variants.

**Decision**: Default to int8 quantized models (`make models` downloads int8).

**Rationale**: ~4x smaller model size (~670MB vs ~2.5GB) with minimal accuracy loss. Significantly reduces download time, disk usage, and memory footprint.

**Consequences**:
- Docker images tagged `latest` use int8
- fp32 available via `make models-fp32` for maximum accuracy
- Both variants use the same code paths

## DD-007: Multi-Stage Docker Build

**Context**: Need containerized deployment.

**Decision**: Multi-stage build with `golang:1.25-bookworm` builder and `debian:bookworm-slim` runtime. Models are embedded in the image.

**Rationale**: Minimal runtime image size. Embedding models avoids runtime downloads and volume mounts for simpler deployment.

**Consequences**:
- Image includes ONNX Runtime 1.21.0
- Separate images for int8 and fp32 model variants
- Health check endpoint (`/health`) included for orchestration
- Exposed port: 5092

## DD-009: Single API Key Authentication

**Context**: The OpenAI-compatible API is exposed over HTTP and needs basic access control.

**Decision**: Support a single API key via the `PARAKEET_API_KEY` environment variable. When set, all `/v1/*` endpoints require `Authorization: Bearer <key>`. When unset, authentication is disabled.

**Rationale**: Simplest possible auth that covers the common case (single deployment, one key). No database, no user management, no token rotation. Matches how most self-hosted AI APIs work (e.g., Ollama, LocalAI). The OpenAI client libraries already send `Authorization: Bearer` headers, so compatibility is automatic.

**Consequences**:
- `/health` endpoint remains unauthenticated (needed for orchestration probes)
- Implemented as `requireAuth()` middleware wrapping `/v1/*` route handlers in `server.go`
- Returns OpenAI-compatible 401 error (`authentication_error`) on invalid/missing key
- Zero overhead when no key is configured (passthrough)

## DD-011: Decoder Session Pool for Concurrent Inference

**Context**: The original implementation created and destroyed an ONNX decoder session inside the TDT decode loop, once per timestep. For a 10-second audio clip with ~150 encoded timesteps, this meant ~150 model loads per request.

**Decision**: Pre-create a pool of `decoderWorker` structs at startup (controlled by `-workers N`). Each worker owns a persistent `ort.AdvancedSession` with pre-allocated, reusable input/output tensors. The decode loop writes directly into the tensor backing data (via `GetData()`) and calls `Run()` without allocating anything.

**Rationale**: Eliminates ~150 session creations per request (the dominant bottleneck). The pool also provides natural backpressure: if all workers are busy, new requests block instead of spawning unconstrained concurrent inference.

**Encoder**: Kept per-request because input shape varies with audio length (dynamic T dimension). The encoder runs once per request — not per timestep — so the overhead is acceptable. The model file is OS page-cached after first load.

**Consequences**:
- `-workers` flag added (default 4); each worker holds ~18MB for decoder + session overhead
- Memory is predictable: `workers × ~670MB` (int8) instead of unbounded concurrent loads
- Throughput: up to `workers` requests processed in parallel
- `decoderWorker.destroy()` called on shutdown to free ORT sessions before `DestroyEnvironment()`
- Graceful shutdown via `http.Server.Shutdown()` ensures all in-flight requests complete before pool is closed

## DD-010: Structured Logging with slog

**Context**: The project used `log.Printf` with ad-hoc `[DEBUG]` prefixes for logging.

**Decision**: Migrate to `log/slog` (stdlib since Go 1.21) with configurable output format (`text` or `json`) via `-log-format` flag and log level controlled by `-log-level`.

**Rationale**: `slog` is stdlib (no new dependencies), provides structured key-value logging, native log levels, and switchable handlers. JSON output is essential for log aggregation in production (ELK, Loki, CloudWatch). Text output stays human-readable for development.

**Consequences**:
- `-log-format` flag added (`text` default, `json` for structured output)
- `-log-level` flag added (`debug`, `info`, `warn`, `error`; default `info`)
- `asr.DebugMode` global derived from `log-level == "debug"`, gates expensive debug logs to avoid unnecessary allocations
- Logger configured once in `main.go:setupLogger()` and set as global default
- All log calls use structured key-value pairs instead of format strings

## DD-008: Minimal Dependencies

**Context**: Go project with potential for many external dependencies.

**Decision**: Only one external Go dependency (`onnxruntime_go`). Everything else uses the standard library.

**Rationale**: Reduces supply chain risk, simplifies builds, and minimizes binary size. Go's stdlib is sufficient for HTTP server, JSON handling, audio processing, and math operations.
