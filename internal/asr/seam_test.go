// SPDX-FileCopyrightText: 2026 Alby Hernández <hola@achetronic.com>
// SPDX-License-Identifier: Apache-2.0

package asr

import (
	"reflect"
	"testing"
)

// dedupSeam is the canary for the seam deduplication rules from issue #18. Each
// case pins one behaviour: exact duplicate within tolerance dropped, duplicate
// outside tolerance kept, text-mismatch collision resolved in favour of the
// earlier window, no collision kept, and the k boundary cases.
func TestDedupSeam(t *testing.T) {
	tests := []struct {
		name     string
		prevTail []decodedToken
		head     []decodedToken
		want     []decodedToken
	}{
		{
			name:     "exact duplicate within tolerance is dropped",
			prevTail: []decodedToken{{id: 5, timestep: 100}},
			head:     []decodedToken{{id: 5, timestep: 101}},
			want:     []decodedToken{},
		},
		{
			name:     "duplicate outside tolerance is kept",
			prevTail: []decodedToken{{id: 5, timestep: 100}},
			head:     []decodedToken{{id: 5, timestep: 110}},
			want:     []decodedToken{{id: 5, timestep: 110}},
		},
		{
			name:     "text mismatch collision keeps the earlier window",
			prevTail: []decodedToken{{id: 5, timestep: 100}},
			head:     []decodedToken{{id: 7, timestep: 101}},
			want:     []decodedToken{},
		},
		{
			name:     "no collision is kept",
			prevTail: []decodedToken{{id: 5, timestep: 100}},
			head:     []decodedToken{{id: 7, timestep: 200}},
			want:     []decodedToken{{id: 7, timestep: 200}},
		},
		{
			name:     "empty previous tail keeps the whole head",
			prevTail: nil,
			head:     []decodedToken{{id: 5, timestep: 100}},
			want:     []decodedToken{{id: 5, timestep: 100}},
		},
		{
			name:     "empty head stays empty",
			prevTail: []decodedToken{{id: 5, timestep: 100}},
			head:     nil,
			want:     nil,
		},
		{
			name:     "mixed head drops colliding tokens and keeps the rest",
			prevTail: []decodedToken{{id: 1, timestep: 50}, {id: 2, timestep: 100}},
			head:     []decodedToken{{id: 2, timestep: 101}, {id: 3, timestep: 102}, {id: 9, timestep: 300}},
			want:     []decodedToken{{id: 9, timestep: 300}},
		},
		{
			name: "only the last seamMaxTokens of the tail are compared",
			prevTail: []decodedToken{
				{id: 1, timestep: 10}, {id: 2, timestep: 20},
				{id: 3, timestep: 30}, {id: 4, timestep: 40},
			},
			// timestep 11 collides with the dropped-from-comparison {1,10} only,
			// so the head token survives.
			head: []decodedToken{{id: 9, timestep: 11}},
			want: []decodedToken{{id: 9, timestep: 11}},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := dedupSeam(tt.prevTail, tt.head)
			// Normalize nil vs empty for the "empty head" case.
			if len(got) == 0 && len(tt.want) == 0 {
				return
			}
			if !reflect.DeepEqual(got, tt.want) {
				t.Fatalf("dedupSeam(%v, %v) = %v, want %v", tt.prevTail, tt.head, got, tt.want)
			}
		})
	}
}
