// SPDX-FileCopyrightText: 2026 Alby Hernández <hola@achetronic.com>
// SPDX-License-Identifier: Apache-2.0

package diarization

import (
	"encoding/binary"
	"os"
	"testing"
)

const (
	framesPerSecond   = 100  // one activity frame per 10 ms
	boundaryTolerance = 0.01 // seconds
)

// TestDiarizeSampleAudio runs the real model over the bundled two-speaker
// audio (A: 0-15 s, 30-45 s; B: 15-30 s, 45-59 s). Opt-in, needs the model:
//
//	DIAR_DIR=./models/diarization/int8 go test ./internal/diarization/ -run TestDiarizeSampleAudio -v
func TestDiarizeSampleAudio(t *testing.T) {
	dir := os.Getenv("DIAR_DIR")
	if dir == "" {
		t.Skip("DIAR_DIR not set, skipping diarization integration test")
	}

	d, err := New(dir, 1)
	if err != nil {
		t.Fatalf("New(%q): %v", dir, err)
	}
	defer d.Close()

	samples := readWAV(t, "../../testdata/diarization/2-speakers.wav")
	act, err := d.Activity(t.Context(), samples)
	if err != nil {
		t.Fatalf("Activity: %v", err)
	}
	if len(act) == 0 {
		t.Fatal("no activity frames")
	}

	// Windows well inside each turn, away from the silence gaps.
	windows := []struct {
		name     string
		from, to float64
	}{
		{"A1", 2, 13},
		{"B1", 18, 28},
		{"A2", 32, 43},
		{"B2", 48, 58},
	}
	dominant := make([]int, len(windows))
	for i, w := range windows {
		dominant[i] = dominantSpeaker(act, w.from, w.to)
		if dominant[i] < 0 {
			t.Fatalf("window %s (%.0f-%.0fs): no speaker above 0.5", w.name, w.from, w.to)
		}
	}

	if dominant[0] != dominant[2] || dominant[1] != dominant[3] {
		t.Fatalf("speaker identity did not persist across chunks: A=%d,%d B=%d,%d",
			dominant[0], dominant[2], dominant[1], dominant[3])
	}
	if dominant[0] == dominant[1] {
		t.Fatalf("both voices attributed to speaker %d", dominant[0])
	}

	segs, err := d.Diarize(t.Context(), samples)
	if err != nil {
		t.Fatalf("Diarize: %v", err)
	}
	if len(segs) == 0 {
		t.Fatal("Diarize returned no segments")
	}
	duration := float64(len(samples)) / float64(d.cfg.SampleRate)
	for _, s := range segs {
		if s.Start < 0 || s.End > duration+boundaryTolerance || s.End < s.Start {
			t.Fatalf("segment out of range: %+v (audio %.2fs)", s, duration)
		}
	}
}

// dominantSpeaker returns the speaker with the most frames above 0.5 inside
// the window, or -1 when nobody is active.
func dominantSpeaker(act [][]float32, from, to float64) int {
	first := int(from * framesPerSecond)
	last := int(to * framesPerSecond)
	if last > len(act) {
		last = len(act)
	}
	votes := make([]int, len(act[0]))
	for f := first; f < last; f++ {
		best, arg := float32(0.5), -1
		for s, p := range act[f] {
			if p > best {
				best, arg = p, s
			}
		}
		if arg >= 0 {
			votes[arg]++
		}
	}
	winner, top := -1, 0
	for s, v := range votes {
		if v > top {
			winner, top = s, v
		}
	}
	return winner
}

// readWAV decodes a 16 kHz mono 16-bit PCM WAV file into float samples.
func readWAV(t *testing.T, path string) []float32 {
	t.Helper()
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read %s: %v", path, err)
	}
	for i := 12; i+8 <= len(data); {
		id := string(data[i : i+4])
		size := int(binary.LittleEndian.Uint32(data[i+4 : i+8]))
		if id == "data" {
			raw := data[i+8 : i+8+size]
			out := make([]float32, len(raw)/2)
			for j := range out {
				out[j] = float32(int16(binary.LittleEndian.Uint16(raw[j*2:]))) / 32768
			}
			return out
		}
		i += 8 + size + size%2
	}
	t.Fatalf("no data chunk in %s", path)
	return nil
}
