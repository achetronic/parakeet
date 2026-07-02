// SPDX-FileCopyrightText: 2026 Alby Hernández <hola@achetronic.com>
// SPDX-License-Identifier: Apache-2.0

package asr

import (
	"errors"
	"reflect"
	"testing"
)

// planChunks must return a single full-coverage window when the audio fits.
func TestPlanChunks_SingleWindowWhenAudioFits(t *testing.T) {
	tests := []struct {
		name  string
		total int64
	}{
		{"well under chunk", 1000},
		{"exactly chunk", 4000},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := planChunks(tt.total, 4000, 500)

			want := []chunkWindow{{start: 0, end: tt.total, emitStart: 0, emitEnd: tt.total}}
			if !reflect.DeepEqual(got, want) {
				t.Fatalf("planChunks(%d) = %+v, want %+v", tt.total, got, want)
			}
		})
	}
}

// planChunks must lay out the known windows and emit ranges for a hand-checked
// case. This is the canary: a wrong midpoint or stride changes these numbers.
func TestPlanChunks_DeterministicLayout(t *testing.T) {
	got := planChunks(100, 40, 10)

	want := []chunkWindow{
		{start: 0, end: 40, emitStart: 0, emitEnd: 35},
		{start: 30, end: 70, emitStart: 35, emitEnd: 65},
		{start: 60, end: 100, emitStart: 65, emitEnd: 100},
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("planChunks(100,40,10) = %+v, want %+v", got, want)
	}
}

// planChunks must produce emit ranges that tile [0,total) with no gap or
// overlap, so every frame is transcribed by exactly one window.
func TestPlanChunks_EmitRangesTileTimeline(t *testing.T) {
	cases := []struct {
		name          string
		total         int64
		chunkFrames   int64
		overlapFrames int64
	}{
		{"even multiple", 90, 40, 10},
		{"one over chunk", 4001, 4000, 500},
		{"large overlap", 100, 40, 35},
		{"no overlap", 100, 40, 0},
		{"many windows", 100000, 30000, 1500},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			windows := planChunks(tc.total, tc.chunkFrames, tc.overlapFrames)

			if len(windows) == 0 {
				t.Fatal("no windows produced")
			}
			if windows[0].emitStart != 0 {
				t.Errorf("first emitStart = %d, want 0", windows[0].emitStart)
			}
			if last := windows[len(windows)-1]; last.emitEnd != tc.total {
				t.Errorf("last emitEnd = %d, want %d", last.emitEnd, tc.total)
			}
			for i, w := range windows {
				if w.emitStart > w.emitEnd {
					t.Errorf("window %d has emitStart %d > emitEnd %d", i, w.emitStart, w.emitEnd)
				}
				if w.start > w.emitStart || w.emitEnd > w.end {
					t.Errorf("window %d emit range [%d,%d) not inside window [%d,%d)", i, w.emitStart, w.emitEnd, w.start, w.end)
				}
				if w.end-w.start > tc.chunkFrames {
					t.Errorf("window %d spans %d frames, exceeds chunk %d", i, w.end-w.start, tc.chunkFrames)
				}
				if i > 0 && windows[i-1].emitEnd != w.emitStart {
					t.Errorf("gap/overlap between window %d emitEnd %d and window %d emitStart %d", i-1, windows[i-1].emitEnd, i, w.emitStart)
				}
			}
		})
	}
}

// planChunks must share exactly overlapFrames between consecutive windows.
func TestPlanChunks_ConsecutiveWindowsShareOverlap(t *testing.T) {
	windows := planChunks(100000, 30000, 1500)

	for i := 1; i < len(windows); i++ {
		// Overlap is where the previous window's end meets this one's start.
		overlap := windows[i-1].end - windows[i].start
		// The final window is truncated at total, so only assert on full ones.
		if windows[i].end-windows[i].start == 30000 && overlap != 1500 {
			t.Errorf("window %d overlap = %d, want 1500", i, overlap)
		}
	}
}

func TestValidateChunking(t *testing.T) {
	const subsampling = 8
	tests := []struct {
		name          string
		chunkFrames   int64
		overlapFrames int64
		wantErr       bool
	}{
		{"valid", 30000, 1500, false},
		{"zero overlap valid", 30000, 0, false},
		{"zero chunk rejected", 0, 0, true},
		{"negative overlap rejected", 30000, -1, true},
		{"overlap equals chunk rejected", 30000, 30000, true},
		{"overlap exceeds chunk rejected", 1000, 2000, true},
		{"exceeds model limit rejected", 40000, 1500, true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := validateChunking(tt.chunkFrames, tt.overlapFrames, subsampling)

			if (err != nil) != tt.wantErr {
				t.Fatalf("validateChunking(%d,%d) err = %v, wantErr %v", tt.chunkFrames, tt.overlapFrames, err, tt.wantErr)
			}
		})
	}
}

func TestPlanForAudio(t *testing.T) {
	const (
		subsampling = 8
		chunk       = 30000
		overlap     = 1500
		// modelMaxEncoderFrames is 5000, so 5000*8 = 40000 mel frames is the
		// single-pass ceiling.
		underLimit = 40000
		overLimit  = 40008
	)
	t.Run("long audio off, short audio, single window", func(t *testing.T) {
		plan, err := planForAudio(underLimit, chunk, overlap, subsampling, false)

		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if len(plan) != 1 || plan[0].start != 0 || plan[0].end != underLimit {
			t.Fatalf("want one full window, got %+v", plan)
		}
	})

	t.Run("long audio off, long audio, rejected", func(t *testing.T) {
		_, err := planForAudio(overLimit, chunk, overlap, subsampling, false)

		if !errors.Is(err, ErrAudioTooLong) {
			t.Fatalf("want ErrAudioTooLong, got %v", err)
		}
	})

	t.Run("long audio on, long audio, chunked", func(t *testing.T) {
		plan, err := planForAudio(overLimit*3, chunk, overlap, subsampling, true)

		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if len(plan) < 2 {
			t.Fatalf("want multiple windows for long audio, got %d", len(plan))
		}
	})

	t.Run("long audio on, short audio, single window", func(t *testing.T) {
		plan, err := planForAudio(1000, chunk, overlap, subsampling, true)

		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if len(plan) != 1 {
			t.Fatalf("want one window, got %d", len(plan))
		}
	})
}

// planChunksWithBoundaries must honour an oracle's boundary when it is inside
// the overlap and clamp it into the legal range otherwise, while keeping the
// tiling invariant for ANY oracle output.
func TestPlanChunksWithBoundaries_HonoursAndClampsOracle(t *testing.T) {
	// Windows for planChunks(100,40,10): starts 0,30,60 ends 40,70,100.
	// Overlaps: [30,40) mid 35, and [60,70) mid 65.
	t.Run("in-range boundaries honoured", func(t *testing.T) {
		oracle := funcOracle{decide: func(r overlapRegion) (int64, bool) {
			if r.start == 30 {
				return 32, true
			}
			return 68, true
		}}
		got := planChunksWithBoundaries(100, 40, 10, oracle)
		want := []chunkWindow{
			{start: 0, end: 40, emitStart: 0, emitEnd: 32},
			{start: 30, end: 70, emitStart: 32, emitEnd: 68},
			{start: 60, end: 100, emitStart: 68, emitEnd: 100},
		}
		if !reflect.DeepEqual(got, want) {
			t.Fatalf("got %+v, want %+v", got, want)
		}
	})

	t.Run("out-of-range boundaries clamped into overlap", func(t *testing.T) {
		oracle := funcOracle{decide: func(r overlapRegion) (int64, bool) {
			if r.start == 30 {
				return -1000, true // below overlapStart -> clamp to 30
			}
			return 1000, true // above overlapEnd -> clamp to 70
		}}
		got := planChunksWithBoundaries(100, 40, 10, oracle)
		want := []chunkWindow{
			{start: 0, end: 40, emitStart: 0, emitEnd: 30},
			{start: 30, end: 70, emitStart: 30, emitEnd: 70},
			{start: 60, end: 100, emitStart: 70, emitEnd: 100},
		}
		if !reflect.DeepEqual(got, want) {
			t.Fatalf("got %+v, want %+v", got, want)
		}
	})
}

// The tiling invariant must hold for arbitrary and adversarial oracles: the emit
// ranges tile [0,total) with no gap or overlap and stay inside their windows,
// no matter what the oracle returns.
func TestPlanChunksWithBoundaries_TilingInvariantArbitraryOracle(t *testing.T) {
	oracles := map[string]boundaryOracle{
		"nil (midpoint)":    nil,
		"always zero":       funcOracle{decide: func(overlapRegion) (int64, bool) { return 0, true }},
		"always huge":       funcOracle{decide: func(overlapRegion) (int64, bool) { return 1 << 40, true }},
		"always negative":   funcOracle{decide: func(overlapRegion) (int64, bool) { return -1 << 40, true }},
		"always decline":    funcOracle{decide: func(overlapRegion) (int64, bool) { return 0, false }},
		"overlap start-ish": funcOracle{decide: func(r overlapRegion) (int64, bool) { return r.start + 1, true }},
		"overlap end-ish":   funcOracle{decide: func(r overlapRegion) (int64, bool) { return r.end - 1, true }},
	}
	cases := []struct {
		name          string
		total         int64
		chunkFrames   int64
		overlapFrames int64
	}{
		{"even multiple", 90, 40, 10},
		{"one over chunk", 4001, 4000, 500},
		{"large overlap", 100, 40, 35},
		{"no overlap", 100, 40, 0},
		{"many windows", 100000, 30000, 1500},
	}
	for oname, oracle := range oracles {
		for _, tc := range cases {
			t.Run(oname+"/"+tc.name, func(t *testing.T) {
				windows := planChunksWithBoundaries(tc.total, tc.chunkFrames, tc.overlapFrames, oracle)
				if len(windows) == 0 {
					t.Fatal("no windows produced")
				}
				if windows[0].emitStart != 0 {
					t.Errorf("first emitStart = %d, want 0", windows[0].emitStart)
				}
				if last := windows[len(windows)-1]; last.emitEnd != tc.total {
					t.Errorf("last emitEnd = %d, want %d", last.emitEnd, tc.total)
				}
				for i, w := range windows {
					if w.emitStart > w.emitEnd {
						t.Errorf("window %d has emitStart %d > emitEnd %d", i, w.emitStart, w.emitEnd)
					}
					if w.start > w.emitStart || w.emitEnd > w.end {
						t.Errorf("window %d emit range [%d,%d) not inside window [%d,%d)", i, w.emitStart, w.emitEnd, w.start, w.end)
					}
					if i > 0 && windows[i-1].emitEnd != w.emitStart {
						t.Errorf("gap/overlap between window %d emitEnd %d and window %d emitStart %d", i-1, windows[i-1].emitEnd, i, w.emitStart)
					}
				}
			})
		}
	}
}

func TestMelToEncoderFrame(t *testing.T) {
	tests := []struct {
		name        string
		melOffset   int64
		subsampling int64
		want        int64
	}{
		{"exact multiple", 800, 8, 100},
		{"rounds down", 807, 8, 100},
		{"zero offset", 0, 8, 0},
		{"zero subsampling passes through", 42, 0, 42},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := melToEncoderFrame(tt.melOffset, tt.subsampling); got != tt.want {
				t.Fatalf("melToEncoderFrame(%d,%d) = %d, want %d", tt.melOffset, tt.subsampling, got, tt.want)
			}
		})
	}
}
