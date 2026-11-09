# Design Decisions

Architectural and design decisions made in this project.

## DD-001: OpenAI Whisper-Compatible API

**Context**: The server needs an HTTP API for audio transcription.

**Decision**: Mirror OpenAI's Whisper endpoints (`/v1/audio/transcriptions`, `/v1/audio/translations`, `/v1/models`).

**Rationale**: Drop-in replacement for existing Whisper clients.

**Consequences**:

- `model`, `prompt`, `temperature` accepted but ignored (one model, greedy decoding).
- Translation delegates to transcription (Parakeet is English-focused).
- Errors follow OpenAI's shape.

## DD-002: NVIDIA Parakeet TDT 0.6B Model

**Context**: Need an ASR model that balances accuracy and resource use.

**Decision**: NVIDIA Parakeet TDT 0.6B: Conformer encoder + Token-and-Duration Transducer (TDT) decoder.

**Rationale**: ~670MB int8 with strong accuracy; TDT predicts tokens and durations, so greedy decoding works without beam search.

**Architecture**:

- Encoder: Conformer, 1024-dim, 8x subsampling.
- Decoder: 8193 vocab (8192 + blank), 5 duration classes, LSTM state 2x640.
- Vocab: SentencePiece, `▁` word boundary, `<blk>` blank at 8192.

## DD-003: ONNX Runtime for Inference

**Context**: Need a runtime for the model in Go.

**Decision**: ONNX Runtime via `onnxruntime_go` v1.30.1 (CPU-first; GPU added in DD-013).

**Rationale**: Portable format, no CGo complexity, vendor-neutral.

**Consequences**:

- ONNX Runtime 1.25.x installed separately; `ONNXRUNTIME_LIB` overrides the path.
- Tensors destroyed manually (no GC).
- ~2GB RAM int8, ~6GB fp32.

## DD-004: WAV-First Audio Input

> Superseded in scope by DD-012 (ffmpeg fallback). Kept for history.

**Context**: Need to accept audio files for transcription.

**Decision**: Parse WAV in pure Go; delegate everything else to an optional external ffmpeg (DD-012).

**Rationale**: No dependencies for the common case.

**Consequences**:

- Content detection by magic bytes (RIFF/WAVE).
- Non-WAV without ffmpeg -> HTTP 400 (`ErrUnsupportedAudio`).
- 8/16/24/32-bit PCM and 32-bit float WAV supported.

## DD-005: Pure Go Audio Processing

**Context**: Need mel-spectrogram features and resampling.

**Decision**: FFT, mel filterbank, windowing, and resampling all in Go stdlib.

**Rationale**: Zero external DSP deps; NeMo-compatible defaults (128 mels, 512 FFT, Hann).

**Consequences**:

- Radix-2 Cooley-Tukey FFT, linear resampling, NeMo-style mean/variance normalization.

## DD-006: Int8 Quantized Models by Default

**Context**: The model ships in fp32 and int8.

**Decision**: Default to int8 (`make model-download`).

**Rationale**: ~4x smaller (~670MB vs ~2.5GB), minimal accuracy loss.

**Consequences**:

- `latest` images use int8; fp32 via `make model-download-asr-fp32`. Same code paths.

## DD-007: Multi-Stage Docker Build

**Context**: Need containerized deployment.

**Decision**: Multi-stage build (`golang:1.25-bookworm` -> `debian:bookworm-slim`), models embedded.

**Rationale**: Minimal runtime image; no runtime downloads or volume mounts.

**Consequences**:

- Ships ONNX Runtime 1.25.1, separate int8/fp32 images, port 5092, `/health`.

## DD-009: Single API Key Authentication

**Context**: The API is exposed over HTTP and needs basic access control.

**Decision**: Optional single key via `PARAKEET_API_KEY`; when set, all `/v1/*` require `Authorization: Bearer <key>`.

**Rationale**: Simplest auth for self-hosted; OpenAI clients already send the header.

**Consequences**:

- `/health` stays open; `requireAuth()` middleware; 401 `authentication_error` on bad key; zero overhead when unset.

## DD-011: Decoder Session Pool

**Context**: The original decode loop built a session per timestep (~150 loads per 10s clip).

**Decision**: Pre-create a pool of `decoderWorker` (one per `-workers`), each with a persistent session and reusable tensors; decode writes into the tensor backing data and calls `Run()` with no allocation.

**Rationale**: Eliminates the per-timestep session cost; the pool caps concurrent inference.

> The shared-encoder assumption below this entry was replaced in DD-019 (per-worker encoder on GPU).

**Consequences**:

- `-workers` flag (default 4); `decoderWorker.destroy()` on shutdown before `DestroyEnvironment()`.
- Graceful shutdown waits for in-flight requests.

## DD-010: Structured Logging with slog

**Context**: Ad-hoc `log.Printf` with `[DEBUG]` prefixes.

**Decision**: `log/slog`, configurable via `-log-level` and `-log-format`.

**Rationale**: Stdlib, structured key-value, JSON for production aggregation, text for dev.

**Consequences**: `asr.DebugMode` derived from `log-level == "debug"`.

## DD-008: Minimal Dependencies

**Context**: Keep the supply chain small.

**Decision**: One external Go dependency (`onnxruntime_go`); everything else stdlib.

**Rationale**: Less risk, simpler builds, smaller binary.

## DD-012: ffmpeg Fallback for Non-WAV Audio

**Context**: DD-004 rejected everything but WAV, but clients upload MP3/WebM/M4A routinely. A prior attempt (PR #5) shared temp-file paths across requests, had no timeout/capture, and mapped every failure to 500.

**Decision**: Optional ffmpeg converter in `internal/audio/ffmpeg.go`.

**Properties**:

- Detection by content (RIFF magic), not extension.
- Binary resolved once at startup; missing -> non-WAV gets 400 (`ErrUnsupportedAudio`), no crash.
- Per-request unique temp files (concurrency-safe).
- `exec.CommandContext` with `-ffmpeg-timeout` (60s); stderr trimmed into the error.
- Conversion failures wrapped in `ErrUnsupportedAudio` -> 400; anything else 500.

**Consequences**: Docker ships ffmpeg; `-workers` bounds converter parallelism too.

## DD-013: Opt-In GPU Inference

> Note: the flag was renamed to `-gpu-devices` and the shared GPU encoder replaced by per-worker sessions in DD-019.

**Context**: Inference was CPU-only. ORT supports CUDA and the graph runs well on it.

**Decision**: `-gpu cpu|cuda` (env `PARAKEET_GPU`); default CPU unchanged. An unknown provider or a broken CUDA setup fails at startup (no silent fallback).

**CUDA tuning** (still in effect):

- `cudnn_conv_algo_search: HEURISTIC` — avoids ORT's EXHAUSTIVE benchmark that can OOM the encoder on long audio.
- `arena_extend_strategy: kSameAsRequested` — avoids power-of-two arena growth.

**fp32 by default on GPU**: the CUDA image ships fp32 (int8 quantization targets CPU integer kernels).

**Consequences**:

- Dedicated `Dockerfile.cuda` (CUDA base + GPU ORT build, fp32, linux/amd64 only), with `make docker-build-cuda`/`docker-run-cuda` and a `docker-cuda` release job.
- CPU images and CPU mode are unaffected.
- Known limit: single-pass encoder means peak memory scales with audio length (chunked encoding is the future fix).
- Other EPs (TensorRT, ROCm, ...) out of scope; add them in `buildSessionOptionsForDevice`.

## DD-014: VAD-Aware Chunk Boundaries + Seam Dedup

**Context**: Long-audio (PR #21) splits audio into overlapping windows and cuts each overlap at its MIDPOINT, which often lands mid-word, so words get emitted twice or dropped (issue #18).

**Decision**: Move the cut onto silence (three-layer cascade), plus an always-on seam token dedup as a backstop. Both act only inside existing overlaps (no global segmentation).

**Boundary cascade** (`boundary.go`, interface-based):

1. `vadBoundaryOracle`: runs Silero VAD over the overlap, picks the center of the longest run of silence (threshold 0.4, a constant not a flag). Falls through if there is no silence.
2. `melEnergyBoundaryOracle`: quietest smoothed mel-energy frame. Always decides when it has features.
3. `midpointBoundaryOracle`: the old midpoint. Always decides, so it is never worse than before.

The result is clamped into the legal overlap range, preserving the tiling invariant.

**Seam dedup** (`seam.go`, always on): drop a leading token of window i+1 when its timestep is within ~3 frames of window i's last tokens; window i wins collisions (its LSTM is already warmed up). Streaming buffers a handful of tokens per seam so dedup runs before the head reaches the client.

**Rejected**: LCS/sequence stitching (fragile, needs a similarity threshold, throws away timing); full-file VAD segmentation (changes what the model sees — out of mission).

**Consequences**:

- New `boundary.go`, `vad.go`, `seam.go`; `chunker.go` gains oracle-aware planning; `transcriber.go` threads absolute timesteps.
- `silero_vad.onnx` (MIT, pinned v6.2.1) lives in `--models`; missing -> warn and degrade to mel energy.
- Flags: `-disable-vad-based-chunking`, `-disable-mel-based-chunking`, `-vad-model-path`.
- Known limit: loud music has no quiet frame, so it degrades to roughly midpoint quality.

## DD-015: Nemotron-3-Diarization

**Context**: Need "who spoke when" alongside ASR.

**Decision**: `Nemotron-3-Diarization` (100M, 8 speakers, openmdw-1.1) via the ONNX export, int8 by default (`model_quantized.onnx`) with fp32 available.

**Rationale**: Language-agnostic, permissive license, fits ORT. Precision follows the Parakeet model.

**Consequences**:

- Streaming chunked encoder with a cache; AOSC/FIFO speaker cache in `config.json`, implemented in Go.
- Own `internal/diarization/` package (own feature extraction, preemphasis 0.97).

## DD-016: Diarization + ASR Merged by Word Timestamps

**Context**: Diarization gives per-frame speaker activity; Parakeet gives word timestamps. They must be combined.

**Decision**: Transcribe fully, diarize to per-frame activity, and attribute each word to the speaker with the highest summed activity over its span.

**Rationale**: Full-audio ASR keeps context; per-frame attribution never drops a word at segment edges.

**Consequences**:

- The merge lives at the server layer (`internal/server/merge.go`).
- `Diarizer.Activity()` exposes per-frame sigmoid probabilities.
- A `speaker` field is added per turn (opt-in `diarize=true`).

## DD-017: Per-Window Mel Normalization for Long Audio

**Context**: Long-audio fed each window a slice of mel features normalized over the WHOLE utterance; on long files those stats sit far from what the encoder saw in training, so it silently dropped whole utterances mid-window.

**Decision**: Normalize each window over its own frames (`mel.Normalize`) instead of globally. Short (single-window) audio is unchanged.

**Result**: long-audio WER dropped to the model's short-clip baseline (human 5.37% -> 3.42% on the eval set).

**Consequences**:

- `Filterbank.Extract` leaves normalization to the caller (`normalize=false`); `transcribeSamples` normalizes per window (~3MB copy each).
- Measurement harness kept as `internal/asr/seameval_test.go` (`-tags=seameval`).

## DD-018: Real Timed Segments From Decoder Token Positions

**Context**: `verbose_json` returned one fake segment and `duration` was derived from byte count (wrong for compressed input).

**Decision**: Build segments from the timesteps the decoder already produces, and measure duration from the decoded waveform. Reporting only — the model sees the same audio.

**Timing**: one encoder frame = `subsampling * hop / sampleRate` = 0.08s. Segments close on gap >= 8 frames, sentence end, 220 tokens, or 30s.

**Consequences**:

- `internal/asr/segments.go`; `decodedToken` gains `logprob`; `Filterbank` gains `SampleRate()`.
- `avg_logprob`, `compression_ratio`, `seek` reported; `temperature`/`no_speech_prob` stay 0 (greedy TDT has neither).
- Streaming unchanged (SSE carries text deltas, not segments).

## DD-019: Concurrent Transcription Support

**Context**: A server serves many clients. One-at-a-time or unbounded hurts throughput and, on GPU, correctness: the old pool (DD-011) only bounded the decoder, so concurrent requests raced the shared encoder on GPU.

**Decision**: Process transcriptions concurrently, bounded by `-workers`, on CPU and GPU, including multiple GPUs / MIG slices. The design is documented in `.agents/CONCURRENCY.md`.

**Consequences**:

- `-workers` bounds the whole inference pipeline, not just the decoder.
- On GPU each worker owns its own encoder+decoder pair, so no ONNX session is shared across requests.
- `-gpu-devices "0,1,2"` (env `PARAKEET_GPU_DEVICES`) replaces `-gpu-device`.

## DD-020: Docker-Free Model Download With crane

**Context**: `model-download-asr-*` pulled model images with `docker pull` + `docker cp`, so users without Docker could not fetch models (issue #44).

**Decision**: Pull and unpack the GHCR model images with `crane export` (single static binary from google/go-containerregistry), downloaded on first use into `./bin` by the `crane` Make target.

**Rationale**: `crane export IMAGE - | tar -x` streams the flattened filesystem, so no Docker daemon and no full-image tarball on disk. One binary, official GitHub release, Linux and macOS on amd64/arm64.

**Consequences**:

- `make model-download-asr-int8` / `-fp32` now only need `curl`, `tar` and `bash`; Docker stays optional (still used to build/push images).
- Models land directly in `models/asr/{int8,fp32}` via `tar --strip-components=1 models`.
- crane is pinned to `CRANE_VERSION` and cached in `./bin`; `make clean` removes it.