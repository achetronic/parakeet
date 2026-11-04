// SPDX-FileCopyrightText: 2026 Alby Hernández <hola@achetronic.com>
// SPDX-License-Identifier: Apache-2.0

package server

import "parakeet/internal/asr"

// merge attributes each word to the speaker with the highest summed activity
// over the word's time span, then groups consecutive words of the same speaker
// into turns. It uses the raw per-frame activity rather than the thresholded
// segments, so words at segment edges are not lost and overlap is resolved by
// the actual probabilities (DD-016 "highest activity").
func merge(words []asr.Word, activity [][]float32, frameDuration float64) []SpeakerTurn {
	var turns []SpeakerTurn
	for _, w := range words {
		spk := dominantSpeaker(activity, w.Start, w.End, frameDuration)
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

// dominantSpeaker returns the index of the speaker with the highest summed
// activity over [start-pad, end+pad]. The pad absorbs the boundary jitter
// between ASR word timestamps and diarization frames.
func dominantSpeaker(activity [][]float32, start, end, frameDuration float64) int {
	const pad = 0.1 // seconds

	if len(activity) == 0 || len(activity[0]) == 0 {
		return 0
	}
	numSpeakers := len(activity[0])

	fs := int((start - pad) / frameDuration)
	fe := int((end + pad) / frameDuration)
	if fs < 0 {
		fs = 0
	}
	if fe > len(activity) {
		fe = len(activity)
	}
	if fs >= fe {
		f := int(start / frameDuration)
		if f < 0 {
			f = 0
		}
		if f >= len(activity) {
			f = len(activity) - 1
		}
		fs, fe = f, f+1
	}

	sums := make([]float32, numSpeakers)
	for f := fs; f < fe; f++ {
		for s := 0; s < numSpeakers; s++ {
			sums[s] += activity[f][s]
		}
	}

	spk := 0
	for s := 1; s < numSpeakers; s++ {
		if sums[s] > sums[spk] {
			spk = s
		}
	}
	return spk
}
