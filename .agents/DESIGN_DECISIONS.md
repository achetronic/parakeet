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

**Rationale**: ONNX is a portable, vendor-neutral format. CPU inference avoids GPU dependency, simplifying deployment. The Go bindings (`onnxruntime_go v1.30.1`) provide direct integration without CGo complexity.

**Consequences**:

- ONNX Runtime library (1.25.x) must be installed separately on the host
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

- Image includes ONNX Runtime 1.25.1
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

**Encoder**: A single long-lived session reused across requests. Input shape varies with audio length (dynamic T dimension), so freshly shaped tensors are passed to each `Run()` rather than rebuilding the session. ORT `Run()` is thread-safe on a shared session, so this is safe under the concurrent worker model. See **DD-013**, where the GPU path makes reusing the session necessary.

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

## DD-013: Opt-In GPU Inference via ONNX Runtime Execution Providers

**Context**: Inference was CPU-only (DD-003). ONNX Runtime already supports GPU acceleration through execution providers (EPs), and the Parakeet graph runs well on the CUDA EP. We want optional GPU inference without changing or risking the CPU default that every existing deployment relies on.

**Decision**: Add an opt-in execution-provider selection. `-gpu cpu|cuda` (env `PARAKEET_GPU`) chooses the provider and `-gpu-device N` (env `PARAKEET_GPU_DEVICE`) chooses the device index. Default `cpu` is byte-for-byte the previous code path. The provider is parsed once (`asr.ParseProvider`) and turned into `*ort.SessionOptions` by `asr.buildSessionOptions`, which is passed to both the encoder session and every decoder worker session.

**Rationale**: A single switch keeps the surface minimal while leaving room for future EPs. Centralizing EP setup in `buildSessionOptions` means the encoder and decoder pool are configured identically, and the CPU path returns `(nil, nil)` options so it is provably unchanged.

**Fail loud, don't fall back**: An unknown provider, or a CUDA EP that fails to initialize, returns an error at startup (server refuses to start) rather than silently degrading to CPU. Silent fallback would hide a misconfigured GPU box behind slow CPU inference — exactly the kind of pipeline failure that should be loud. A non-integer `PARAKEET_GPU_DEVICE` is treated as unset after a warning, so a typo can't quietly select the wrong device.

**CUDA provider tuning**: The CUDA EP is configured with two non-default options, both to bound VRAM use on the single-pass encoder:

- `cudnn_conv_algo_search: HEURISTIC` — the ORT default (`EXHAUSTIVE`) benchmarks every cuDNN convolution algorithm on first run and can reserve many GB of workspace for one Conv, OOMing the encoder on longer audio. `HEURISTIC` picks a good algorithm without that up-front allocation.
- `arena_extend_strategy: kSameAsRequested` — grows the GPU arena by exactly what is requested instead of the default power-of-two steps, which would compound the above.

**fp32 by default on GPU**: The CUDA image ships fp32 models. int8 quantization targets CPU integer kernels and offers little benefit on GPU, where fp32 runs natively with full accuracy and VRAM is the binding constraint rather than host RAM.

**Memory ownership**: `buildSessionOptions` returns a `*ort.SessionOptions` the caller owns. ORT copies the options into each session at creation time, so `NewTranscriber` defers a single `Destroy()` that runs only after the encoder and all decoder workers exist. The shared long-lived encoder session (DD-011) is a prerequisite: rebuilding it per request would repeatedly pay CUDA context/provider setup and negate the GPU win.

**Consequences**:

- `-gpu`/`-gpu-device` flags and `PARAKEET_GPU`/`PARAKEET_GPU_DEVICE` env vars added; flag-over-env precedence via `envOr`/`envInt` in `main.go`.
- A dedicated `*-cuda` Docker image (`Dockerfile.cuda`) built on an NVIDIA CUDA base with a GPU build of ONNX Runtime, fp32 models, `linux/amd64` only. Matching `make docker-build-cuda` / `make docker-run-cuda` (the latter needs `--gpus all` / nvidia-container-toolkit) and a `docker-cuda` release job publishing `*-cuda` tags to ghcr.io.
- CPU images, CPU mode, and the int8/fp32 CPU images are unaffected.
- **Known limitation**: the encoder runs in a single pass, so peak memory scales with audio length. On GPU this is bounded by VRAM — very long inputs can exceed device memory and fail with an ORT allocation error. Chunked/streaming encoding is the future fix.
- Other accelerators (TensorRT, ROCm, DirectML, OpenVINO) are out of scope; `buildSessionOptions` is the single place to add them.

## DD-014: VAD-Aware Chunk Boundaries and Seam Token Dedup

**Context**: Long-audio mode (PR #21) covers audio longer than the model's 5000 encoder-frame limit with overlapping windows and splits ownership of each overlap at its arithmetic MIDPOINT (`chunker.go`). Because the midpoint is blind to content it often lands mid-word. Each window timestamps tokens slightly differently, so a word straddling the boundary is emitted twice (duplicates like "to to") or by neither window (drops like "for., and how"). Issue #18 reproduces this on a ~30 minute file.

**Decision**: Move the boundary onto silence with a three-layer cascade, and add an always-on seam-level token dedup as a second safety net. Both operate only within the existing overlaps; the mission stays faithful to Parakeet TDT (no global segmentation, no silence skipping).

### Boundary selection cascade (interface-based oracles)

A `boundaryOracle` interface (`boundary.go`) picks the mel-frame split inside an overlap. The three implementations are alternative strategies for the same contract, tried in order by `chainBoundaryOracle`:

1. `vadBoundaryOracle`: runs Silero VAD over the overlap's waveform and picks the CENTER of the LONGEST run of 32 ms windows whose speech probability is below `vadSilenceThreshold` (0.4, a tuned constant in the agreed 0.35-0.5 band, NOT a flag). Silero's state is reset per overlap, so its first ~1 s is warmup (`vadWarmupWindows`); a boundary after the warmup is preferred, falling back to the warmup region only if no later silence exists. Declines (falls through) when nothing is below the threshold.
2. `melEnergyBoundaryOracle`: smooths per-frame energy from the already-extracted mel features (`melSmoothingFrames` ~150 ms) and returns the quietest frame in the overlap. Always decides when it has features, so it is the robust fallback when the VAD is disabled or its model is missing.
3. `midpointBoundaryOracle`: the arithmetic midpoint, the pre-#18 behaviour. Always decides, so the result is never worse than before.

`planChunksWithBoundaries` takes the oracle and CLAMPS whatever it returns into the legal overlap range, keeping the tiling invariant (`emitEnd[i] == emitStart[i+1]`, first `emitStart` 0, last `emitEnd` total, each emit range inside its window) for ANY oracle output. `planChunks` is now the nil-oracle (pure-midpoint) case, so the existing canary tests are unchanged.

**Why an interface**: the three layers are genuinely interchangeable implementations of one decision, and the cascade is just "try each until one decides". An interface makes each layer independently unit-testable from synthetic probability/energy arrays and lets the disable flags drop layers by simply not adding them to the chain.

**Why VAD only on overlaps**: the overlaps are short (15 s each by default), so the VAD cost is bounded and proportional to the number of seams, not the file length. Running Silero over the whole file would be the first step towards VAD segmentation, which this project deliberately rejects.

### Silero VAD integration

- Model file `silero_vad.onnx` lives in the `--models` directory like the other models, downloaded by the Makefile and Docker builds (NOT embedded in the binary). Pinned to release **v6.2.1** (MIT) with sha256 `1a153a22f4509e292a94e67d6f9b85e8deb25b4988682b7e174c65279d8788e3`, verified with `sha256sum -c` at download time. NOTE: at v6.2.1 the ONNX file is at `src/silero_vad/data/silero_vad.onnx`; the older `files/silero_vad.onnx` path no longer exists.
- One shared `DynamicAdvancedSession`, same pattern as the encoder (DD-011). Silero is stateful, but its recurrent state travels through the `state`/`stateN` TENSORS per call, not inside the session, so a single session is safe under concurrency as long as each request keeps its own `vadState`. The loop feeds 512-sample (32 ms) windows sequentially, prepending a 64-sample context and carrying the state between calls; v5+ uses the combined `state` tensor plus an `sr` sample-rate input, verified against the pinned model.
- **Missing model is not fatal**: `slog.Warn` once at startup ("VAD model not found, chunk boundaries fall back to mel energy") and degrade to the mel layer. Any other load error is fatal so a corrupt model surfaces loudly.
- **Concurrency caveat**: like the encoder, the VAD session runs OUTSIDE the decoder worker pool, so its concurrency is bounded by the number of in-flight HTTP requests, not by `-workers`.

### Seam token dedup (always on, no flag)

Operating on emitted tokens tagged with their ABSOLUTE encoder-frame timesteps (`decodedToken`, `seam.go`), `dedupSeam` drops a leading token of window i+1 when its timestep is within `seamTimestepToleranceFrames` (3 frames, ~240 ms) of any of window i's last `seamMaxTokens` (3) tokens. One rule covers both #18 failure modes: same text at the same position is a duplicate; different text at the same position is a collision that window i WINS (its LSTM reaches the seam fully warmed up while window i+1 is still warming). A duplicate further apart than the tolerance is kept. All tolerances are named constants with comments, not flags.

**Streaming**: `TranscribeStream` emits deltas as tokens are produced. To let dedup run before window i+1's head reaches the client, `tdtDecode` BUFFERS at most the first `seamMaxTokens` owned tokens of each window after the first, resolves them through the deduper, streams the survivors in order, then streams the rest as it decodes. Buffering is minimal (a handful of tokens per seam) and streaming order is preserved.

### Rejected alternatives

- **LCS / sequence stitching** of overlapping transcripts: fragile on repeated words, needs a similarity threshold that is itself a tuning knob, and operates on text after the timing information that actually identifies the seam has been thrown away. Placing the boundary on silence removes the ambiguity at the source; the timestep-based dedup is a cheap, deterministic backstop.
- **Full-file VAD segmentation**: would skip silence and re-segment globally, changing what audio Parakeet sees and breaking faithfulness to the model. Out of scope by mission.

### Known limitation

Loud or dynamic music has no quiet frame, so the mel-energy layer degrades to roughly midpoint quality; the VAD layer is the robust answer there because it distinguishes speech from non-speech energy rather than loud from quiet. With the VAD disabled AND music present, boundaries fall back to the midpoint, i.e. the pre-#18 behaviour.

**Configuration surface** (env vars come free via `applyEnvDefaults`):

- `--disable-vad-based-chunking` (bool, default false): drops layer 1.
- `--disable-mel-based-chunking` (bool, default false): drops layer 2.
- `--vad-model-path` (string, default `silero_vad.onnx` inside `--models`).

**Consequences**:

- New files: `internal/asr/boundary.go` (oracle stack), `internal/asr/vad.go` (Silero session), `internal/asr/seam.go` (dedup). `chunker.go` gains `planChunksWithBoundaries`/`planForAudioWithBoundaries`; `transcriber.go` threads absolute timesteps, the seam buffer, and the per-request oracle chain.
- `silero_vad.onnx` is a new required-for-VAD model file; its absence is a graceful degrade, not a failure.
- Reference reproduction material (the issue #18 MP3 plus `.srt`/`.vtt`/`.txt`) lives in `testdata/reference/`, with a build-tag-gated Go seam inspector (`-tags=seaminspect`) that prints transcribed vs reference text around every seam. No Python, no network.
