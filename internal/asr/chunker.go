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
// larger than chunkFrames, each sharing overlapFrames with its neighbours.
//
// The overlap gives the encoder acoustic context and the decoder LSTM time to
// warm up before its owned region begins. Ownership of the shared overlap is
// split at its midpoint: the earlier window emits up to the midpoint, the later
// window from the midpoint on, so every frame is emitted by exactly one window.
//
// When the audio fits in a single window (total <= chunkFrames), it returns one
// window covering everything, which reproduces the non-chunked behaviour.
//
// Callers must pass chunkFrames > overlapFrames >= 0 and total > 0.
func planChunks(total, chunkFrames, overlapFrames int64) []chunkWindow {
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
	// end, and every interior boundary sits at the midpoint of the overlap
	// shared by the two windows straddling it.
	for i := range windows {
		if i == 0 {
			windows[i].emitStart = 0
		} else {
			windows[i].emitStart = windows[i-1].emitEnd
		}

		if i == len(windows)-1 {
			windows[i].emitEnd = total
		} else {
			// Midpoint of the overlap between window i and i+1.
			windows[i].emitEnd = (windows[i+1].start + windows[i].end) / 2
		}
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
	if longAudio {
		return planChunks(total, chunkFrames, overlapFrames), nil
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
