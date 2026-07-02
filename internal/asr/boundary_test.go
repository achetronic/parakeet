// SPDX-FileCopyrightText: 2026 Alby Hernández <hola@achetronic.com>
// SPDX-License-Identifier: Apache-2.0

package asr

import "testing"

// funcOracle is a test boundaryOracle whose decision is supplied by a closure,
// so tests can drive arbitrary and adversarial oracle behaviour.
type funcOracle struct {
	id     string
	calls  *int
	decide func(r overlapRegion) (int64, bool)
}

func (f funcOracle) name() string { return f.id }

func (f funcOracle) boundary(r overlapRegion) (int64, bool) {
	if f.calls != nil {
		*f.calls++
	}
	return f.decide(r)
}

// The midpoint oracle always decides and returns the arithmetic midpoint, so it
// can terminate the cascade.
func TestMidpointBoundaryOracle_AlwaysDecidesMidpoint(t *testing.T) {
	frame, ok := midpointBoundaryOracle{}.boundary(overlapRegion{start: 30, end: 40, midpoint: 35})
	if !ok || frame != 35 {
		t.Fatalf("midpoint oracle = (%d,%v), want (35,true)", frame, ok)
	}
}

// The cascade returns the first oracle that decides and does not consult later
// oracles once one has decided.
func TestChainBoundaryOracle_FallthroughOrder(t *testing.T) {
	var c1, c2, c3 int
	chain := chainBoundaryOracle{oracles: []boundaryOracle{
		funcOracle{id: "a", calls: &c1, decide: func(overlapRegion) (int64, bool) { return 0, false }},
		funcOracle{id: "b", calls: &c2, decide: func(overlapRegion) (int64, bool) { return 7, true }},
		funcOracle{id: "c", calls: &c3, decide: func(overlapRegion) (int64, bool) { return 9, true }},
	}}

	frame, ok := chain.boundary(overlapRegion{start: 0, end: 20, midpoint: 10})
	if !ok || frame != 7 {
		t.Fatalf("chain = (%d,%v), want (7,true)", frame, ok)
	}
	if c1 != 1 || c2 != 1 {
		t.Fatalf("expected first two oracles consulted once, got c1=%d c2=%d", c1, c2)
	}
	if c3 != 0 {
		t.Fatalf("third oracle must not be consulted after a decision, got c3=%d", c3)
	}
}

// A cascade in which every oracle declines returns ok=false, so the planner
// falls back to the midpoint itself.
func TestChainBoundaryOracle_AllDecline(t *testing.T) {
	chain := chainBoundaryOracle{oracles: []boundaryOracle{
		funcOracle{id: "a", decide: func(overlapRegion) (int64, bool) { return 0, false }},
		funcOracle{id: "b", decide: func(overlapRegion) (int64, bool) { return 0, false }},
	}}
	if _, ok := chain.boundary(overlapRegion{start: 0, end: 10, midpoint: 5}); ok {
		t.Fatal("chain must decline when all oracles decline")
	}
}

// Canary: longestSubThresholdCenter must pick the centre of the longest run of
// windows below the threshold, honour the warmup preference, and decline when no
// window is below the threshold. A regression in the selection changes these.
func TestLongestSubThresholdCenter(t *testing.T) {
	const th = 0.4
	tests := []struct {
		name   string
		probs  []float32
		warmup int
		want   int
		wantOK bool
	}{
		{
			name:   "single run centre",
			probs:  []float32{0.9, 0.1, 0.1, 0.1, 0.9},
			warmup: 0,
			want:   2, // run [1,3], centre 2
			wantOK: true,
		},
		{
			name:   "longest of two runs",
			probs:  []float32{0.1, 0.1, 0.9, 0.1, 0.1, 0.1, 0.9},
			warmup: 0,
			want:   4, // longer run [3,5], centre 4
			wantOK: true,
		},
		{
			name:   "tie keeps earliest run",
			probs:  []float32{0.1, 0.1, 0.9, 0.1, 0.1},
			warmup: 0,
			want:   0, // both runs length 2; earliest [0,1], centre 0
			wantOK: true,
		},
		{
			name:   "warmup skips early run for later one",
			probs:  []float32{0.1, 0.1, 0.1, 0.9, 0.1, 0.1},
			warmup: 3,
			want:   4, // only [4,5] eligible after warmup, centre 4
			wantOK: true,
		},
		{
			name:   "warmup falls back when only silence is in warmup",
			probs:  []float32{0.1, 0.1, 0.1, 0.9, 0.9, 0.9},
			warmup: 3,
			want:   1, // nothing after warmup; fall back to whole region, run [0,2] centre 1
			wantOK: true,
		},
		{
			name:   "no window below threshold",
			probs:  []float32{0.9, 0.8, 0.7},
			warmup: 0,
			want:   0,
			wantOK: false,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, ok := longestSubThresholdCenter(tt.probs, th, tt.warmup)
			if ok != tt.wantOK || (ok && got != tt.want) {
				t.Fatalf("longestSubThresholdCenter(%v, warmup=%d) = (%d,%v), want (%d,%v)",
					tt.probs, tt.warmup, got, ok, tt.want, tt.wantOK)
			}
		})
	}
}

func TestFrameEnergies(t *testing.T) {
	features := [][]float32{
		{1, 2, 3}, // 6
		{0, 0, 0}, // 0
		{-1, -1},  // -2
	}
	got := frameEnergies(features)
	want := []float64{6, 0, -2}
	if len(got) != len(want) {
		t.Fatalf("len = %d, want %d", len(got), len(want))
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("frameEnergies[%d] = %v, want %v", i, got[i], want[i])
		}
	}
}

func TestSmoothEnergies(t *testing.T) {
	// Window 3, centred moving average with edge clamping.
	in := []float64{0, 3, 6, 9}
	got := smoothEnergies(in, 3)
	want := []float64{
		(0 + 3) / 2.0,     // 1.5
		(0 + 3 + 6) / 3.0, // 3
		(3 + 6 + 9) / 3.0, // 6
		(6 + 9) / 2.0,     // 7.5
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("smoothEnergies[%d] = %v, want %v", i, got[i], want[i])
		}
	}

	// Window <= 1 returns the input unchanged.
	if got := smoothEnergies(in, 1); &got[0] != &in[0] {
		t.Fatal("smoothEnergies with window 1 must return the input slice unchanged")
	}
}

func TestArgMinInRange(t *testing.T) {
	values := []float64{5, 2, 8, 1, 9, 1}
	tests := []struct {
		name       string
		start, end int
		want       int
	}{
		{"full range, first minimum wins", 0, 6, 3},
		{"sub range", 0, 3, 1},
		{"range excludes global min", 0, 3, 1},
		{"out of bounds clamped", 4, 100, 5},
		{"empty range returns start", 2, 2, 2},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := argMinInRange(values, tt.start, tt.end); got != tt.want {
				t.Fatalf("argMinInRange(%d,%d) = %d, want %d", tt.start, tt.end, got, tt.want)
			}
		})
	}
}

// The mel-energy oracle must place the boundary inside the quiet valley within
// the overlap, and always decide when it has features.
func TestMelEnergyBoundaryOracle(t *testing.T) {
	// 80 frames of loud audio with a quiet valley at frames [40,50).
	features := make([][]float32, 80)
	for i := range features {
		v := float32(10)
		if i >= 40 && i < 50 {
			v = 0
		}
		features[i] = []float32{v}
	}
	oracle := newMelEnergyBoundaryOracle(features)

	frame, ok := oracle.boundary(overlapRegion{start: 30, end: 60, midpoint: 45})
	if !ok {
		t.Fatal("mel-energy oracle must decide when it has features")
	}
	if frame < 40 || frame >= 50 {
		t.Fatalf("boundary %d not inside the quiet valley [40,50)", frame)
	}

	// No features: declines so the cascade falls through.
	empty := newMelEnergyBoundaryOracle(nil)
	if _, ok := empty.boundary(overlapRegion{start: 0, end: 10, midpoint: 5}); ok {
		t.Fatal("mel-energy oracle must decline without features")
	}
}
