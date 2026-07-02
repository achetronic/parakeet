# TODO

Pending tasks and improvements for the project.

## Audio Format Support

- [x] **ffmpeg-based audio conversion** — Implemented in `internal/asr/ffmpeg.go`. WAV is parsed in-process (magic-byte detection); any other format is transcoded via an external `ffmpeg` binary. Configurable with `-ffmpeg`, `-ffmpeg-path`, `-ffmpeg-timeout`. See DD-012 for rationale.

## API Completeness

- [ ] **Implement `prompt` parameter** — Currently accepted but ignored in the transcription endpoint. Could be used for vocabulary biasing or context priming.
- [ ] **Implement `temperature` parameter** — Currently accepted but ignored. Would require switching from greedy to sampled decoding in `tdtDecode()`.
- [ ] **Proper translation support** — The `/v1/audio/translations` endpoint currently delegates to transcription. Parakeet is English-focused, so true translation would require a different model or pipeline.

## Performance

- [x] **VAD-aware chunk boundaries + seam dedup**: Fixes chunk-seam hallucinations (issue #18) in long-audio mode. Overlap ownership is split on silence via a VAD -> mel-energy -> midpoint cascade, and a seam-level token dedup removes duplicated/colliding tokens. Toggle layers with `-disable-vad-based-chunking` / `-disable-mel-based-chunking`; VAD model path via `-vad-model-path`. See DD-014.
- [x] **GPU inference** — Implemented via ONNX Runtime execution providers. Opt in with `-gpu cuda` / `-gpu-device N`; a dedicated `*-cuda` Docker image ships the GPU build. CPU remains the default. See DD-013. TensorRT and other accelerators remain out of scope (extend `buildSessionOptions`).
- [ ] **Batch inference support** — Current implementation processes one audio file at a time. Batching multiple requests could improve throughput under load.
- [ ] **Higher quality resampling** — `audio.go:resample()` uses linear interpolation. Sinc-based or polyphase resampling would improve audio quality.

## Testing

- [ ] **Expand test coverage** — Add integration tests for the full transcription pipeline and edge cases (very short audio, max length audio, various WAV formats).
