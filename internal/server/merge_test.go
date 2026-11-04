// SPDX-FileCopyrightText: 2026 Alby Hernández <hola@achetronic.com>
// SPDX-License-Identifier: Apache-2.0

package server

import (
	"reflect"
	"testing"

	"parakeet/internal/asr"
	"parakeet/internal/diarization"
)

func TestMerge(t *testing.T) {
	words := []asr.Word{
		{Text: "hello", Start: 0.1, End: 0.4},
		{Text: "world", Start: 0.5, End: 0.8},
		{Text: "how", Start: 1.0, End: 1.2},
		{Text: "are", Start: 1.3, End: 1.5},
		{Text: "you", Start: 1.6, End: 1.8},
	}
	segs := []diarization.Segment{
		{Start: 0.0, End: 0.9, Speaker: 0},
		{Start: 0.95, End: 2.0, Speaker: 1},
	}
	want := []SpeakerTurn{
		{Speaker: 0, Start: 0.1, End: 0.8, Text: "hello world"},
		{Speaker: 1, Start: 1.0, End: 1.8, Text: "how are you"},
	}
	got := merge(words, segs)
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("merge = %+v, want %+v", got, want)
	}
}

func TestMergeOverlapPrefersMostRecentSegment(t *testing.T) {
	words := []asr.Word{{Text: "hi", Start: 0.5, End: 0.6}}
	segs := []diarization.Segment{
		{Start: 0.0, End: 0.7, Speaker: 0},
		{Start: 0.4, End: 1.0, Speaker: 1},
	}
	got := merge(words, segs)
	if len(got) != 1 || got[0].Speaker != 1 {
		t.Fatalf("merge = %+v, want speaker 1", got)
	}
}

func TestMergeSkipsWordsOutsideSegments(t *testing.T) {
	words := []asr.Word{{Text: "hi", Start: 5.0, End: 5.2}}
	segs := []diarization.Segment{{Start: 0.0, End: 1.0, Speaker: 0}}
	if got := merge(words, segs); len(got) != 0 {
		t.Fatalf("expected no turns, got %+v", got)
	}
}

func TestMergeToleratesWordStartingBeforeSegment(t *testing.T) {
	// A word that starts just before the segment's edge (within mergeTolerance)
	// is still attributed to that speaker.
	words := []asr.Word{{Text: "In", Start: 0.48, End: 0.72}}
	segs := []diarization.Segment{{Start: 0.68, End: 7.49, Speaker: 0}}
	got := merge(words, segs)
	if len(got) != 1 || got[0].Speaker != 0 {
		t.Fatalf("merge = %+v, want one speaker-0 turn", got)
	}
}
