// SPDX-FileCopyrightText: 2026 Alby Hernández <hola@achetronic.com>
// SPDX-License-Identifier: Apache-2.0

package asr

import (
	"errors"
	"fmt"
)

// ErrAudioTooLong is returned when long-audio mode is off and the input exceeds
// what the model can process in a single pass.
var ErrAudioTooLong = errors.New("audio too long for a single pass; enable long-audio mode")

const (
	// DefaultChunkSeconds and DefaultChunkOverlapSeconds are the out-of-the-box
	// window and overlap sizes when a caller leaves them unset.
	DefaultChunkSeconds        = 300
	DefaultChunkOverlapSeconds = 15

	// modelMaxEncoderFrames is the longest encoder-frame sequence the exported
	// ONNX model accepts: its positional-encoding table is [1, 9999, 1024],
	// centred at 5000, so beyond 5000 frames the relative-position slice goes
	// out of range and the self-attention Add crashes on a broadcast mismatch.
	modelMaxEncoderFrames int64 = 5000

	// chunkSafetyMarginEncoderFrames keeps windows clear of the hard model
	// limit so rounding at the subsampling boundary can never reach it.
	chunkSafetyMarginEncoderFrames int64 = 200
)

// chunkWindow describes one analysis window over the mel feature sequence.
//
// All fields are absolute mel-frame indices into the full utterance, half-open
// [start, end). The encoder is run over [start, end); to avoid duplicating
// speech in the overlap between adjacent windows, only tokens whose position
// falls in [emitStart, emitEnd) are kept. Adjacent emit ranges tile the whole
// timeline with no gaps or overlaps: emitEnd of one window equals emitStart of
// the next.
type chunkWindow struct {
	start     int64
	end       int64
	emitStart int64
	emitEnd   int64
}

// planChunks splits a mel sequence of total frames into overlapping windows no
// larger than chunkFrames, each sharing overlapFrames with its neighbours,
// splitting each overlap at its arithmetic midpoint. It is the pure-midpoint
// case of planChunksWithBoundaries and is kept for the tests and callers that
// do not need a smarter boundary.
//
// Callers must pass chunkFrames > overlapFrames >= 0 and total > 0.
func planChunks(total, chunkFrames, overlapFrames int64) []chunkWindow {
	return planChunksWithBoundaries(total, chunkFrames, overlapFrames, nil)
}

// planChunksWithBoundaries splits a mel sequence of total frames into overlapping
// windows no larger than chunkFrames, each sharing overlapFrames with its
// neighbours, and asks oracle where to split the ownership of each interior
// overlap.
//
// The overlap gives the encoder acoustic context and the decoder LSTM time to
// warm up before its owned region begins. Ownership of the shared overlap is
// split at a single mel frame: the earlier window emits up to it, the later
// window from it on, so every frame is emitted by exactly one window. A nil
// oracle (or one that declines) falls back to the arithmetic midpoint, which
// reproduces the pre-issue-#18 behaviour.
//
// Whatever the oracle returns, the boundary is clamped so the emit ranges still
// tile [0, total) with no gap or overlap and stay inside each window: the first
// window emits from 0, the last emits to total, emitStart is non-decreasing, and
// emitEnd[i] == emitStart[i+1]. This invariant holds for ANY oracle output.
//
// When the audio fits in a single window (total <= chunkFrames), it returns one
// window covering everything, which reproduces the non-chunked behaviour.
//
// Callers must pass chunkFrames > overlapFrames >= 0 and total > 0.
func planChunksWithBoundaries(total, chunkFrames, overlapFrames int64, oracle boundaryOracle) []chunkWindow {
	if total <= chunkFrames {
		return []chunkWindow{{start: 0, end: total, emitStart: 0, emitEnd: total}}
	}

	stride := chunkFrames - overlapFrames

	var windows []chunkWindow
	for start := int64(0); start < total; start += stride {
		end := start + chunkFrames
		if end >= total {
			end = total
		}
		windows = append(windows, chunkWindow{start: start, end: end})
		if end == total {
			break
		}
	}

	// Assign emit ranges. The first window owns from 0, the last owns to the
	// end, and every interior boundary is chosen by the oracle inside the
	// overlap shared by the two windows straddling it, then clamped so the
	// tiling invariant holds regardless of what the oracle returns.
	prevEmitEnd := int64(0)
	for i := range windows {
		windows[i].emitStart = prevEmitEnd

		if i == len(windows)-1 {
			windows[i].emitEnd = total
			prevEmitEnd = total
			continue
		}

		overlapStart := windows[i+1].start
		overlapEnd := windows[i].end
		midpoint := (overlapStart + overlapEnd) / 2

		boundary := midpoint
		if oracle != nil {
			if frame, ok := oracle.boundary(overlapRegion{
				start:    overlapStart,
				end:      overlapEnd,
				midpoint: midpoint,
			}); ok {
				boundary = frame
			}
		}

		// Clamp into the legal overlap range, then keep emitStart monotonic so a
		// pathological oracle can never make emitStart exceed emitEnd. Because
		// window ends are non-decreasing, prevEmitEnd <= overlapEnd always, so
		// the final boundary stays inside [emitStart, windows[i].end].
		if boundary < overlapStart {
			boundary = overlapStart
		}
		if boundary > overlapEnd {
			boundary = overlapEnd
		}
		if boundary < windows[i].emitStart {
			boundary = windows[i].emitStart
		}

		windows[i].emitEnd = boundary
		prevEmitEnd = boundary
	}

	return windows
}

// melToEncoderFrame converts a mel-frame offset to its encoder-frame index under
// the given subsampling factor. The encoder collapses subsampling mel frames
// into one, so the mapping is integer division.
func melToEncoderFrame(melOffset, subsampling int64) int64 {
	if subsampling <= 0 {
		return melOffset
	}
	return melOffset / subsampling
}

// planForAudio decides how to cover a mel sequence of total frames. With long
// audio enabled it splits into overlapping windows (planChunks). With it off it
// returns a single full-coverage window, or ErrAudioTooLong when the input would
// overrun the model's single-pass limit, so the caller fails cleanly instead of
// letting the encoder crash on an out-of-range positional-encoding slice.
func planForAudio(total, chunkFrames, overlapFrames, subsampling int64, longAudio bool) ([]chunkWindow, error) {
	return planForAudioWithBoundaries(total, chunkFrames, overlapFrames, subsampling, longAudio, nil)
}

// planForAudioWithBoundaries is planForAudio with a boundary oracle: when long
// audio is on it uses the oracle to place each interior overlap boundary
// (falling back to the midpoint when the oracle declines or is nil). The
// single-window and ErrAudioTooLong paths are unchanged.
func planForAudioWithBoundaries(total, chunkFrames, overlapFrames, subsampling int64, longAudio bool, oracle boundaryOracle) ([]chunkWindow, error) {
	if longAudio {
		return planChunksWithBoundaries(total, chunkFrames, overlapFrames, oracle), nil
	}
	if melToEncoderFrame(total, subsampling) > modelMaxEncoderFrames {
		return nil, ErrAudioTooLong
	}
	return []chunkWindow{{start: 0, end: total, emitStart: 0, emitEnd: total}}, nil
}

// validateChunking rejects window sizes that would break planChunks or overrun
// the model's positional-encoding limit. Sizes are in mel frames.
func validateChunking(chunkFrames, overlapFrames, subsampling int64) error {
	if chunkFrames <= 0 {
		return fmt.Errorf("chunk size must be positive, got %d frames", chunkFrames)
	}
	if overlapFrames < 0 {
		return fmt.Errorf("chunk overlap must not be negative, got %d frames", overlapFrames)
	}
	if overlapFrames >= chunkFrames {
		return fmt.Errorf("chunk overlap (%d frames) must be smaller than chunk size (%d frames)", overlapFrames, chunkFrames)
	}
	maxFrames := (modelMaxEncoderFrames - chunkSafetyMarginEncoderFrames) * subsampling
	if chunkFrames > maxFrames {
		return fmt.Errorf("chunk size (%d frames) exceeds the model-safe maximum (%d frames)", chunkFrames, maxFrames)
	}
	return nil
}
