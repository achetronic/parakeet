// SPDX-FileCopyrightText: 2026 Alby Hernández <hola@achetronic.com>
// SPDX-License-Identifier: Apache-2.0

package server

import (
	"parakeet/internal/asr"
	"parakeet/internal/diarization"
)

// merge assigns each word to the speaker active at its start time and groups
// consecutive words of the same speaker into turns. In overlapping speech the
// most recently started segment wins; segments carry no per-frame score, so
// this approximates "highest activity".
func merge(words []asr.Word, segs []diarization.Segment) []SpeakerTurn {
	var turns []SpeakerTurn
	for _, w := range words {
		spk, ok := speakerAt(segs, w.Start)
		if !ok {
			continue
		}
		if n := len(turns); n > 0 && turns[n-1].Speaker == spk {
			turns[n-1].Text += " " + w.Text
			if w.End > turns[n-1].End {
				turns[n-1].End = w.End
			}
			continue
		}
		turns = append(turns, SpeakerTurn{Speaker: spk, Start: w.Start, End: w.End, Text: w.Text})
	}
	return turns
}

// mergeTolerance absorbs the boundary jitter between ASR word timestamps and
// diarization segment edges (the two models segment the same audio slightly
// differently).
const mergeTolerance = 0.25 // seconds

// speakerAt returns the speaker of the segment containing t, preferring the
// segment that started most recently when several overlap. A word starting
// slightly before a segment's edge (within mergeTolerance) still counts.
func speakerAt(segs []diarization.Segment, t float64) (int, bool) {
	spk := 0
	found := false
	bestStart := -1.0
	for _, s := range segs {
		if s.Start-mergeTolerance <= t && t < s.End && (!found || s.Start > bestStart) {
			spk = s.Speaker
			found = true
			bestStart = s.Start
		}
	}
	return spk, found
}
