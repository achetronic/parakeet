// SPDX-FileCopyrightText: 2026 Alby Hernández <hola@achetronic.com>
// SPDX-License-Identifier: Apache-2.0

package asr

import "log/slog"

// This file implements the chunk-boundary selection stack described in DD-014.
//
// When long-audio mode splits an utterance into overlapping windows, the shared
// overlap between two neighbours must be split at a single mel-frame index that
// decides which window emits which tokens. A blind arithmetic midpoint can land
// in the middle of a word, so the word is timestamped slightly differently by
// each window and ends up emitted twice or dropped (issue #18). The oracles
// below try to move that split onto silence instead.

const (
	// vadSilenceThreshold is the Silero speech-probability level below which a
	// window counts as silence for boundary selection. Tuned within the agreed
	// 0.35-0.5 band: low enough to ignore breaths and room tone, high enough to
	// still find gaps between words in continuous speech. Not a flag on purpose.
	vadSilenceThreshold float32 = 0.4

	// vadWarmupWindows is how many leading 32 ms windows of an overlap the VAD
	// treats as warmup. Silero's recurrent state is reset at the start of every
	// overlap, so its first ~1 s of probabilities are unreliable; we prefer a
	// boundary after the warmup and only fall back to the warmup region when no
	// silence is found later. 16000 Hz / 512 samples ~= 31 windows per second.
	vadWarmupWindows = int(vadSampleRate) / vadWindowSamples

	// melSmoothingFrames is the moving-average width (in mel frames, 10 ms each)
	// applied to per-frame energy before picking the quietest point. ~150 ms
	// smooths out single-frame dips so the minimum lands in a genuine pause
	// rather than on a transient glottal closure inside a word.
	melSmoothingFrames = 15
)

// overlapRegion describes, in absolute mel-frame coordinates, the region shared
// by window i and window i+1 that a boundaryOracle must split. midpoint is the
// historical arithmetic split, provided so an oracle can bias towards it.
type overlapRegion struct {
	start    int64 // first mel frame of the overlap (inclusive)
	end      int64 // one past the last mel frame of the overlap (exclusive)
	midpoint int64 // arithmetic midpoint, the pre-issue-#18 boundary
}

// boundaryOracle is the contract for choosing an emission boundary inside an
// overlap region. The implementations below are alternative strategies for the
// same contract, tried as a cascade (see chainBoundaryOracle):
//
//   - vadBoundaryOracle:      centre of the longest Silero-detected silence.
//   - melEnergyBoundaryOracle: quietest smoothed mel-energy frame.
//   - midpointBoundaryOracle:  the arithmetic midpoint (always decides; the
//     final fallback so the result is never worse than before).
//
// boundary returns the chosen split as an absolute mel-frame index and true when
// the oracle can decide, or (_, false) to fall through to the next layer. The
// returned frame is advisory: planChunksWithBoundaries clamps it into the legal
// overlap range so the tiling invariant always holds.
type boundaryOracle interface {
	boundary(r overlapRegion) (int64, bool)
	name() string
}

// midpointBoundaryOracle reproduces the pre-issue-#18 behaviour: split at the
// arithmetic midpoint of the overlap. It always decides, so it is the terminal
// layer of the cascade and guarantees a boundary is always chosen.
type midpointBoundaryOracle struct{}

func (midpointBoundaryOracle) name() string { return "midpoint" }

func (midpointBoundaryOracle) boundary(r overlapRegion) (int64, bool) {
	return r.midpoint, true
}

// melEnergyBoundaryOracle splits the overlap at the quietest point of the
// already-extracted mel features. It smooths per-frame energy once for the whole
// utterance, then returns the minimum inside each overlap. It always yields a
// value when it has features, so it is the robust fallback when the VAD is
// disabled or unavailable. Loud, dynamic music defeats the energy heuristic
// (there is no quiet frame), in which case this degrades to roughly midpoint
// quality; the VAD layer is the robust answer there (see DD-014).
type melEnergyBoundaryOracle struct {
	smoothed []float64
}

func newMelEnergyBoundaryOracle(features [][]float32) *melEnergyBoundaryOracle {
	return &melEnergyBoundaryOracle{
		smoothed: smoothEnergies(frameEnergies(features), melSmoothingFrames),
	}
}

func (o *melEnergyBoundaryOracle) name() string { return "mel-energy" }

func (o *melEnergyBoundaryOracle) boundary(r overlapRegion) (int64, bool) {
	if len(o.smoothed) == 0 || r.end <= r.start {
		return 0, false
	}
	idx := int64(argMinInRange(o.smoothed, int(r.start), int(r.end)))
	return idx, true
}

// vadBoundaryOracle splits the overlap at the centre of the longest run of
// Silero-detected silence. It runs the VAD only over the overlap's waveform (not
// the whole file: this repo stays faithful to Parakeet TDT and never segments or
// skips audio globally), resetting the recurrent state per overlap. When no run
// falls below the silence threshold it falls through to the next layer.
type vadBoundaryOracle struct {
	vad       *sileroVAD
	state     *vadState
	waveform  []float32
	hopLength int64
}

func (o *vadBoundaryOracle) name() string { return "vad" }

func (o *vadBoundaryOracle) boundary(r overlapRegion) (int64, bool) {
	if o.vad == nil || r.end <= r.start {
		return 0, false
	}

	sampleStart := r.start * o.hopLength
	sampleEnd := r.end * o.hopLength
	if sampleStart < 0 {
		sampleStart = 0
	}
	if sampleEnd > int64(len(o.waveform)) {
		sampleEnd = int64(len(o.waveform))
	}
	if sampleEnd-sampleStart < vadWindowSamples {
		return 0, false
	}

	probs := o.vad.speechProbabilities(o.state, o.waveform[sampleStart:sampleEnd])
	if len(probs) == 0 {
		return 0, false
	}

	center, ok := longestSubThresholdCenter(probs, vadSilenceThreshold, vadWarmupWindows)
	if !ok {
		return 0, false
	}

	// Map the chosen window back to the centre sample, then to a mel frame,
	// clamped defensively into the overlap.
	centerSample := sampleStart + int64(center)*vadWindowSamples + vadWindowSamples/2
	frame := centerSample / o.hopLength
	if frame < r.start {
		frame = r.start
	}
	if frame >= r.end {
		frame = r.end - 1
	}
	return frame, true
}

// chainBoundaryOracle tries each oracle in order and returns the first decision,
// implementing the VAD -> mel-energy -> midpoint cascade. A nil or empty chain
// decides nothing, which makes planChunksWithBoundaries fall back to the
// midpoint itself.
type chainBoundaryOracle struct {
	oracles []boundaryOracle
}

func (c chainBoundaryOracle) name() string { return "chain" }

func (c chainBoundaryOracle) boundary(r overlapRegion) (int64, bool) {
	for _, o := range c.oracles {
		if frame, ok := o.boundary(r); ok {
			if DebugMode {
				slog.Debug("chunk boundary chosen",
					"oracle", o.name(),
					"frame", frame,
					"overlapStart", r.start,
					"overlapEnd", r.end,
					"midpoint", r.midpoint,
				)
			}
			return frame, true
		}
	}
	return 0, false
}

// longestSubThresholdCenter returns the window index at the centre of the
// longest run of consecutive windows whose probability is below threshold. It
// prefers a run that starts at or after warmupWindows (Silero's unreliable
// warmup) and only considers the warmup region when no later silence exists. It
// returns false when no window is below threshold at all. Ties on run length
// keep the earliest run, so the selection is deterministic (a canary property).
func longestSubThresholdCenter(probs []float32, threshold float32, warmupWindows int) (int, bool) {
	if c, ok := longestRunCenter(probs, threshold, warmupWindows); ok {
		return c, true
	}
	return longestRunCenter(probs, threshold, 0)
}

func longestRunCenter(probs []float32, threshold float32, from int) (int, bool) {
	if from < 0 {
		from = 0
	}
	bestStart, bestLen := -1, 0
	runStart := -1
	for i := from; i < len(probs); i++ {
		if probs[i] < threshold {
			if runStart < 0 {
				runStart = i
			}
			if runLen := i - runStart + 1; runLen > bestLen {
				bestLen = runLen
				bestStart = runStart
			}
			continue
		}
		runStart = -1
	}
	if bestStart < 0 {
		return 0, false
	}
	return bestStart + (bestLen-1)/2, true
}

// frameEnergies reduces each mel frame to a single loudness proxy by summing its
// (normalized log-mel) bins. Quieter frames sum lower, so the minimum marks the
// best place to cut. Working off the already-extracted features means the
// mel-energy layer costs nothing extra to compute.
func frameEnergies(features [][]float32) []float64 {
	energies := make([]float64, len(features))
	for i, frame := range features {
		var sum float64
		for _, v := range frame {
			sum += float64(v)
		}
		energies[i] = sum
	}
	return energies
}

// smoothEnergies applies a centred moving average of the given width (in frames)
// using a prefix sum, so single-frame dips inside a word do not masquerade as
// pauses. A width <= 1 returns the input unchanged.
func smoothEnergies(energies []float64, window int) []float64 {
	if window <= 1 || len(energies) == 0 {
		return energies
	}
	n := len(energies)
	prefix := make([]float64, n+1)
	for i := 0; i < n; i++ {
		prefix[i+1] = prefix[i] + energies[i]
	}
	half := window / 2
	out := make([]float64, n)
	for i := 0; i < n; i++ {
		lo := i - half
		if lo < 0 {
			lo = 0
		}
		hi := i + half + 1
		if hi > n {
			hi = n
		}
		out[i] = (prefix[hi] - prefix[lo]) / float64(hi-lo)
	}
	return out
}

// argMinInRange returns the index of the smallest value in values[start:end),
// clamping the range into bounds. Ties keep the earliest index, so selection is
// deterministic. When the range is empty it returns start.
func argMinInRange(values []float64, start, end int) int {
	if start < 0 {
		start = 0
	}
	if end > len(values) {
		end = len(values)
	}
	if start >= end {
		return start
	}
	minIdx := start
	for i := start + 1; i < end; i++ {
		if values[i] < values[minIdx] {
			minIdx = i
		}
	}
	return minIdx
}
