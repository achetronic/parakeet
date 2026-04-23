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

## DD-004: WAV-First Audio Input (with ffmpeg fallback)

> **Note**: originally "WAV-Only Audio Input". Superseded in scope by **DD-012** when ffmpeg-backed conversion was introduced; this entry is kept for historical context.

**Context**: The server needs to accept audio files for transcription.

**Decision**: Natively support WAV in pure Go. Delegate any other format to an optional external `ffmpeg` binary (see DD-012).

**Rationale**: WAV parsing is straightforward with no external dependencies. Keeping the fast path in-process preserves the "no external dependencies at runtime" value for the common case while still enabling broad format support when ffmpeg is available.

**Consequences**:

- `loadAudio()` in `transcriber.go` detects WAV by magic bytes (RIFF/WAVE) and parses it in-process.
- Non-WAV input is routed to the ffmpeg converter; when ffmpeg is unavailable the request returns HTTP 400 with `ErrUnsupportedAudio`.
- Supports 8/16/24/32-bit PCM and 32-bit float WAV natively.
- All audio resampled to 16kHz mono internally.
- Minimum audio length: 100ms (1600 samples at 16kHz).

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

## DD-012: ffmpeg-Backed Conversion for Non-WAV Audio

**Context**: The original implementation (DD-004) accepted only WAV input and returned a hard error for anything else. That worked, but forced every client to preprocess audio, which is awkward for a Whisper-compatible API (OpenAI clients upload MP3/WebM/M4A routinely). A previous community attempt (PR #5) added ffmpeg but had three problems: it shared temp-file paths across concurrent requests (breaking DD-011's worker pool guarantees), had no timeout/stderr capture, and mapped any failure to HTTP 500.

**Decision**: Introduce an optional ffmpeg-backed converter encapsulated in `internal/asr/ffmpeg.go`.

Key properties:

1. **Detection by content, not extension.** `loadAudio` inspects the first 12 bytes of the payload. If it is a `RIFF ... WAVE` header, parse in-process (zero-deps fast path). Otherwise, hand the bytes to the converter.
2. **Startup probe.** The ffmpeg binary is resolved once via `exec.LookPath` when the transcriber is built. If it is missing, the converter is simply `nil`: the server starts normally, logs a warning, and rejects non-WAV uploads with a clear HTTP 400 (`ErrUnsupportedAudio`). No crash, no surprise runtime failure.
3. **Per-request unique temp files.** Each `Convert()` call uses `os.CreateTemp` for both input and output. This is required because DD-011's worker pool allows up to `-workers` concurrent inferences, and each of them may be preceded by a conversion.
4. **Bounded execution.** `exec.CommandContext` with a configurable timeout (`-ffmpeg-timeout`, default 60s). `stderr` is captured and trimmed into the error message so operators can diagnose bad input.
5. **Typed errors.** Conversion failures (bad input, timeout, binary missing) are wrapped in `ErrUnsupportedAudio`. The HTTP handler checks with `errors.Is` and returns `400 invalid_request_error`. Everything else stays as `500 server_error`.

**Configuration surface**:

- `-ffmpeg` (bool, default `true`) — toggles the fallback.
- `-ffmpeg-path` (string, default empty → resolve via `PATH`).
- `-ffmpeg-timeout` (duration, default `60s`).

**Consequences**:

- Binary releases remain self-contained but optionally leverage a system-installed ffmpeg. The Docker image ships with ffmpeg by default.
- `DD-004` is superseded in scope: we still support WAV in pure Go as the fast path, but we no longer reject every other format outright.
- Concurrency semantics of DD-011 are preserved: conversions are independent per request, so `-workers N` continues to bound both decoding _and_ converter parallelism naturally.
- OpenAI-compatibility improves: clients can upload MP3/WebM/M4A directly, matching the behavior of the real Whisper API.
