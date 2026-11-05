// SPDX-FileCopyrightText: 2026 Alby Hernández <hola@achetronic.com>
// SPDX-License-Identifier: Apache-2.0

package asr

import (
	"math"
	"testing"

	"parakeet/internal/mel"
)

func TestWordsFromTokens(t *testing.T) {
	tests := []struct {
		name  string
		vocab map[int]string
		toks  []decodedToken
		want  []Word
	}{
		{
			name: "groups boundary and continuation tokens into one word",
			vocab: map[int]string{
				1: " hello",
				2: "world",
				3: " again",
			},
			toks: []decodedToken{
				{id: 1, timestep: 0, duration: 2},
				{id: 2, timestep: 2, duration: 2},
				{id: 3, timestep: 10, duration: 2},
			},
			want: []Word{
				{Text: "helloworld", Start: 0.0, End: 0.32},
				{Text: "again", Start: 0.8, End: 0.96},
			},
		},
		{
			name: "punctuation attaches to the previous word",
			vocab: map[int]string{
				1: " hi",
				2: ",",
			},
			toks: []decodedToken{
				{id: 1, timestep: 0, duration: 1},
				{id: 2, timestep: 1, duration: 0},
			},
			want: []Word{
				{Text: "hi,", Start: 0.0, End: 0.16},
			},
		},
		{
			name: "skips special tokens",
			vocab: map[int]string{
				1: " hi",
				2: "<blk>",
			},
			toks: []decodedToken{
				{id: 1, timestep: 0, duration: 1},
				{id: 2, timestep: 1, duration: 1},
			},
			want: []Word{
				{Text: "hi", Start: 0.0, End: 0.08},
			},
		},
		{
			name: "continuation with no leading boundary still emits a word",
			vocab: map[int]string{
				1: "suffix",
			},
			toks: []decodedToken{
				{id: 1, timestep: 0, duration: 1},
			},
			want: []Word{
				{Text: "suffix", Start: 0.0, End: 0.08},
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tr := &Transcriber{
				vocab:  tt.vocab,
				config: Config{SubsamplingFactor: 8},
				mel:    mel.New(128, 16000, 512, 160, 400, 0.97),
			}
			got := tr.wordsFromTokens(tt.toks)
			if len(got) != len(tt.want) {
				t.Fatalf("got %d words, want %d: %+v", len(got), len(tt.want), got)
			}
			for i := range got {
				if got[i].Text != tt.want[i].Text {
					t.Errorf("word %d text = %q, want %q", i, got[i].Text, tt.want[i].Text)
				}
				if !almostEqual(got[i].Start, tt.want[i].Start) || !almostEqual(got[i].End, tt.want[i].End) {
					t.Errorf("word %d times = (%v, %v), want (%v, %v)", i, got[i].Start, got[i].End, tt.want[i].Start, tt.want[i].End)
				}
			}
		})
	}
}

func almostEqual(a, b float64) bool {
	return math.Abs(a-b) < 1e-6
}
