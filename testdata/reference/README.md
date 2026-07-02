# Reference audio for chunk-seam validation

This directory holds the exact audio and reference transcripts used to reproduce
and validate the fix for issue #18 (duplicated or dropped words at the seams
between overlapping windows in long-audio mode).

## Provenance

- Source: the YouTube video "Learn Case Interviews in Under 30 minutes", the
  exact file attached to issue #18.
- Original filenames (renamed here to shell-friendly ones):
  - `Learn Case Interviews in Under 30 minutes.mp3` -> `learn-case-interviews.mp3`
  - `Learn Case Interviews in Under 30 minutes.srt` -> `learn-case-interviews.srt`
  - `Learn Case Interviews in Under 30 minutes.vtt` -> `learn-case-interviews.vtt`
  - `Learn Case Interviews in Under 30 minutes.txt` -> `learn-case-interviews.txt`

The MP3 (~41 MB) is committed to the repository on purpose: reproducing the
seam behaviour needs the real long-audio input, and a stable checked-in copy
means the validation does not depend on any network access.

## Files

- `learn-case-interviews.mp3` - the input audio (~30 minutes, needs `-long-audio`).
- `learn-case-interviews.srt` / `.vtt` - timed reference transcripts, used to
  print the expected words around each seam for a side-by-side comparison.
- `learn-case-interviews.txt` - the plain reference transcript.

## Usage

The seam inspector is a build-tag-gated Go test (no Python in this repo). It
transcribes the MP3 through the full pipeline with long-audio enabled, logs the
chosen boundary positions, and prints the transcribed text around each seam next
to the reference `.srt` text for the same time range, so a human can eyeball
every seam quickly. It needs ONNX Runtime and the models locally but no network.

```bash
# Requires the models (make models-int8) and ONNX Runtime on the host.
PARAKEET_MODELS=./models \
PARAKEET_SEAM_AUDIO=./testdata/reference/learn-case-interviews.mp3 \
PARAKEET_SEAM_SRT=./testdata/reference/learn-case-interviews.srt \
go test -tags=seaminspect -run TestSeamInspection -v ./internal/asr/
```

See `.agents/DESIGN_DECISIONS.md` (DD-014) for the boundary-selection design.
