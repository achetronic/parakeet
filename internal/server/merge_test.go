// SPDX-FileCopyrightText: 2026 Alby Hernández <hola@achetronic.com>
// SPDX-License-Identifier: Apache-2.0

package server

import (
	"reflect"
	"testing"

	"parakeet/internal/asr"
)

// activity2 builds a [numFrames][2] activity at 0.1s/frame where speaker 0 is
// active in [s0, e0) frames and speaker 1 in [s1, e1) frames.
func activity2(numFrames, s0, e0, s1, e1 int) [][]float32 {
	act := make([][]float32, numFrames)
	for f := range act {
		row := make([]float32, 2)
		if f >= s0 && f < e0 {
			row[0] = 1
		}
		if f >= s1 && f < e1 {
			row[1] = 1
		}
		act[f] = row
	}
	return act
}

func TestMergeByActivity(t *testing.T) {
	words := []asr.Word{
		{Text: "hello", Start: 0.1, End: 0.4},
		{Text: "world", Start: 0.5, End: 0.8},
		{Text: "hi", Start: 1.1, End: 1.4},
	}
	act := activity2(20, 0, 10, 10, 20) // spk 0 in 0-1s, spk 1 in 1-2s
	got := merge(words, act, 0.1)
	want := []SpeakerTurn{
		{Speaker: 0, Start: 0.1, End: 0.8, Text: "hello world"},
		{Speaker: 1, Start: 1.1, End: 1.4, Text: "hi"},
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("merge = %+v, want %+v", got, want)
	}
}

func TestMergeOverlapHighestActivity(t *testing.T) {
	words := []asr.Word{{Text: "hi", Start: 0.5, End: 0.6}}
	act := make([][]float32, 10)
	for f := range act {
		act[f] = []float32{0.5, 0.8} // speaker 1 consistently louder
	}
	got := merge(words, act, 0.1)
	if len(got) != 1 || got[0].Speaker != 1 {
		t.Fatalf("merge = %+v, want speaker 1", got)
	}
}

func TestMergeBoundaryJitter(t *testing.T) {
	// The word starts at 0.48s but the speaker is only active from 0.5s; the
	// pad absorbs the jitter instead of dropping the word.
	words := []asr.Word{{Text: "In", Start: 0.48, End: 0.72}}
	act := activity2(10, 5, 10, 0, 0)
	got := merge(words, act, 0.1)
	if len(got) != 1 || got[0].Speaker != 0 {
		t.Fatalf("merge = %+v, want speaker 0", got)
	}
}

func TestMergeEmptyActivity(t *testing.T) {
	words := []asr.Word{{Text: "hi", Start: 0.1, End: 0.2}}
	got := merge(words, nil, 0.1)
	if len(got) != 1 || got[0].Speaker != 0 {
		t.Fatalf("merge = %+v, want one speaker-0 turn", got)
	}
}
