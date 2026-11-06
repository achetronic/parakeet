# TODO

Pending tasks and improvements for the project.

## Audio Format Support

- [x] **ffmpeg conversion** — Done. Non-WAV auto-converted via ffmpeg (configurable). See DD-012.

## API Completeness

- [x] **Real timed segments** — Done. Segments from decoder timesteps; duration from the waveform. See DD-018.
- [ ] **`prompt` parameter** — Accepted but ignored; could bias the vocabulary.
- [ ] **`temperature` parameter** — Accepted but ignored; needs sampled decoding.
- [ ] **Translation** — `/v1/audio/translations` just delegates to transcription.

## Performance

- [x] **VAD-aware chunk boundaries + seam dedup** — Done. Silence-aware overlaps; dedup removes duplicate/colliding tokens. See DD-014.
- [x] **GPU inference** — Done. `-gpu cuda` / `-gpu-devices`; CUDA Docker image. See DD-013, DD-019.
- [x] **Concurrent transcriptions** — Done and validated on a real GPU. Worker pool gates the whole pipeline; per-worker encoder on GPU. See DD-019.
- [ ] **Batch inference** — One audio file at a time today; batching could raise throughput.
- [ ] **Better resampling** — `resample()` is linear; sinc/polyphase would sound better.

## Speaker Diarization

- [x] **Speaker diarization** — Done. Nemotron-3 ONNX + word-timestamp merge; `diarize=true`. See DD-015, DD-016.

## Testing

- [ ] **Expand test coverage** — Integration tests: very short, max-length, and various WAV formats.