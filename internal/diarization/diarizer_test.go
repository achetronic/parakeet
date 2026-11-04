// SPDX-FileCopyrightText: 2026 Alby Hernández <hola@achetronic.com>
// SPDX-License-Identifier: Apache-2.0

package diarization

import (
	"reflect"
	"testing"
)

func TestPlanChunks(t *testing.T) {
	got := planChunks(700, 340, 40, 8)
	want := []chunkPlan{
		{melStart: 0, melEnd: 3040, numChunk: 340},
		{melStart: 2720, melEnd: 5600, numChunk: 340},
		{melStart: 5440, melEnd: 5600, numChunk: 20},
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("planChunks = %+v, want %+v", got, want)
	}
}

func TestPlanChunksSingleShort(t *testing.T) {
	got := planChunks(100, 340, 40, 8)
	if len(got) != 1 {
		t.Fatalf("expected 1 chunk, got %d", len(got))
	}
	if got[0].numChunk != 100 || got[0].melStart != 0 || got[0].melEnd != 800 {
		t.Fatalf("unexpected chunk: %+v", got[0])
	}
}

func TestExtractSegments(t *testing.T) {
	// logits at 4 frames x 3 speakers; sigmoid(x)>0.5 <=> x>0.
	logits := [][]float32{
		{1, -1, -1}, // spk 0 active
		{1, 1, -1},  // spk 0 + 1 active (overlap)
		{-1, 1, -1}, // spk 1 active
		{-1, -1, 1}, // spk 2 active
	}
	got := extractSegments(logits, 3, 0.01)
	want := []Segment{
		{Start: 0.00, End: 0.02, Speaker: 0},
		{Start: 0.01, End: 0.03, Speaker: 1},
		{Start: 0.03, End: 0.04, Speaker: 2},
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("extractSegments = %+v, want %+v", got, want)
	}
}

func TestExtractSegmentsEmpty(t *testing.T) {
	if got := extractSegments(nil, 8, 0.01); got != nil {
		t.Fatalf("expected nil for empty logits, got %+v", got)
	}
}
