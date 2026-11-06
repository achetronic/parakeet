# Concurrent Transcriptions

How Parakeet handles several transcriptions at once, on CPU and GPU, including
pods with multiple GPUs or MIG slices. This is the approach, not the build
plan; the concrete changes live in DESIGN_DECISIONS.md (DD-019) and TODO.md.

## How one transcription works

Each request runs through these stages, in order:

1. Read the audio (parse WAV, or call ffmpeg for MP3 and friends).
2. Mel: turn the waveform into sound features. Plain Go math.
3. VAD: find silences to pick cut points. Only for long audio.
4. Encoder: the network reads the features and produces an internal
   representation.
5. Decoder: step by step, with LSTM memory, it emits words until the text is
   done.

## What we share and what we don't

The model weights are loaded once:

- Encoder: one copy, shared, because it is stateless per request.
- VAD: one copy.
- Diarizer: one copy.
- Decoder: one copy per worker in a pool, because the decoder carries state
  from one token to the next.

Everything else (waveform, mel features, tensors, LSTM state) is created and
thrown away per request.

## What is broken today

There is no gate at the front. Every request that arrives runs stages 1 to 4
at the same time, with no limit, because the only semaphore sits at the very
end, at the decoder.

- The pure Go math (mel, VAD) parallelizes fine and is safe, but unbounded.
- The encoder is one shared copy that every request hits at once. On CPU that
  is fine. On GPU it is not: the card cannot run two convolutions with
  different audio lengths on the same session at once, so it corrupts state or
  crashes (the known MIOpen/cuDNN failure).
- Only the decoder is bounded today: N workers, N parallel decodes max.

Everything but the decoder runs in parallel with no control, and on GPU that is
dangerous, not fast.

## The approach: one gate, one inference unit per GPU worker

Put one gate at the front, and one unit of work behind it.

**The gate.** A worker is acquired when a request enters the inference pipeline,
and held through mel, VAD, encoder, and decoder. At most `-workers` requests
are inside at any moment. This bounds every stage, not just the decoder.

**The GPU worker is a full unit.** Instead of sharing one encoder across
requests, each GPU worker owns its own encoder and its own decoder, both on the
same device. The N workers are spread round-robin over the devices you list.
With nothing shared there is nothing to race, so the encoder needs no lock and
runs N ways in parallel.

**CPU keeps a shared encoder.** On CPU, a shared session is safe to run
concurrently, and host RAM is more expensive than VRAM, so one copy stays
shared and N decoders sit in the pool.

### By provider

| Piece | CPU | GPU |
|---|---|---|
| Worker pool (the gate) | N | N |
| Encoder | 1 shared, N-way concurrent | N, one private per worker, round-robin over K devices |
| Silero VAD | CPU + mutex | CPU + mutex |
| Diarizer | CPU + semaphore(N) | CPU + semaphore(N) |

Nothing is serialized. On one big GPU this is faster than a shared, serialized
encoder would be; on several GPUs or MIG slices it scales across them.

## Config

- `-gpu-devices "0,1,2"` (env `PARAKEET_GPU_DEVICES`), default `"0"`. The
  devices to use. `"0"` is today's behavior.
- `-workers N`: total concurrent transcriptions, spread across those devices.

MIG slices and physical GPUs give real parallelism, because each is a separate
device with its own memory. Time-slicing shows several "devices" that all map
to one physical GPU: it does not break anything, it just does not speed
anything up. It pays to know which one your platform gives you.

## Memory and VRAM

- CPU: 1 encoder + N decoders + 1 VAD + 1 diarizer, all resident.
- GPU: N encoders + N decoders in VRAM, roughly N x 670 MB for int8 models.
  Size `-workers` to what the GPUs or slices can hold.
- Per-request peak (mel matrix, tensors) goes from unbounded to at most N.

## Trade-offs and limits

- Time-slicing gives no speedup; only physical GPUs or MIG slices do.
- N separate encoder sessions on one physical GPU is the same pattern the N
  decoders already use, with `cudnn_conv_algo_search=HEURISTIC` and
  `arena_extend_strategy=kSameAsRequested` set to avoid the conflicting
  algorithm search. CI has no GPU, so this must be validated on real hardware
  before trusting it: 32 requests, 8 workers, mixed clip lengths.
- Audio reading, ffmpeg, and WAV parsing sit outside the gate. They are I/O
  bound and capped by the 25 MB body limit and `-ffmpeg-timeout`; the gate
  starts where the inference cost starts, at mel.
- `diarize=true` runs after the ASR worker is released, with its own
  semaphore(N), so mixed requests can peak at about 2N.
- SSE streaming holds its worker for the life of the stream; the 30 s write
  deadline already handles slow readers.