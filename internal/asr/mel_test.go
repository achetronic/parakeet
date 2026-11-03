// SPDX-FileCopyrightText: 2026 Alby Hernández <hola@achetronic.com>
// SPDX-License-Identifier: Apache-2.0

package asr

import (
	"math"
	"testing"
)

// nemoTestSignal is a deterministic 1s multi-tone 16 kHz signal used as the
// golden input for the NeMo frontend checks.
func nemoTestSignal() []float32 {
	const n = 16000
	x := make([]float32, n)
	for i := 0; i < n; i++ {
		f := float64(i)
		v := 0.5*math.Sin(2*math.Pi*440*f/16000) +
			0.25*math.Sin(2*math.Pi*1800*f/16000) +
			0.1*math.Sin(2*math.Pi*5200*f/16000)
		x[i] = float32(v)
	}
	return x
}

// TestMelMatchesNemoGolden pins the extractor against values produced by
// onnx-asr's NemoPreprocessorNumpy (itself matching nemo128.onnx to
// ~7e-5) for nemoTestSignal(). If the mel math drifts from NeMo's
// frontend, these values move and the test fails loudly: the multilingual
// drift this guards against is invisible on clean studio audio.
func TestMelMatchesNemoGolden(t *testing.T) {
	m := NewMelFilterbank(128, 16000)
	features := m.Extract(nemoTestSignal())

	if len(features) != 100 {
		t.Fatalf("frame count = %d, want 100 (16000 samples / 160 hop)", len(features))
	}
	for i, frame := range features {
		if len(frame) != 128 {
			t.Fatalf("frame %d has %d features, want 128", i, len(frame))
		}
	}

	golden := []struct {
		frame, bin int
		want       float64
	}{
		{0, 0, 6.681776},
		{0, 63, 9.505524},
		{0, 127, 8.466496},
		{50, 0, -0.612673},
		{50, 64, -0.106060},
		{50, 127, -0.152607},
		{99, 0, 5.455378},
		{99, 64, 0.214308},
		{99, 127, 1.587718},
	}
	const tol = 1e-3
	for _, g := range golden {
		got := float64(features[g.frame][g.bin])
		if math.Abs(got-g.want) > tol {
			t.Errorf("features[%d][%d] = %.6f, want %.6f (tol %g)", g.frame, g.bin, got, g.want, tol)
		}
	}
}

func TestMelNormalization(t *testing.T) {
	m := NewMelFilterbank(128, 16000)
	features := m.Extract(nemoTestSignal())

	n := float64(len(features))
	for bin := 0; bin < 128; bin++ {
		var sum float64
		for _, f := range features {
			sum += float64(f[bin])
		}
		mean := sum / n
		var varSum float64
		for _, f := range features {
			d := float64(f[bin]) - mean
			varSum += d * d
		}
		std := math.Sqrt(varSum / (n - 1))
		if math.Abs(mean) > 1e-3 {
			t.Errorf("bin %d: mean = %f, want ~0", bin, mean)
		}
		if math.Abs(std-1) > 1e-2 {
			t.Errorf("bin %d: std = %f, want ~1", bin, std)
		}
	}
}

func TestMelFrameCount(t *testing.T) {
	m := NewMelFilterbank(128, 16000)
	cases := []struct {
		samples int
		want    int
	}{
		{16000, 100},
		{16001, 100},
		{16159, 100},
		{16160, 101},
		{880032, 5500}, // 55.002s, one of the drift benchmark clips
		{159, 0},
		{160, 1},
	}
	for _, c := range cases {
		x := make([]float32, c.samples)
		got := len(m.Extract(x))
		if got != c.want {
			t.Errorf("Extract(%d samples) frames = %d, want %d", c.samples, got, c.want)
		}
	}
}

func TestSlaneyMelScale(t *testing.T) {
	// Slaney scale is linear below 1 kHz (f_sp = 200/3 Hz per mel) and
	// log above; mel_to_hz must invert hz_to_mel at both ends.
	for _, hz := range []float64{0, 100, 999, 1000, 2000, 7999} {
		back := slaneyMelToHz(slaneyHzToMel(hz))
		if math.Abs(back-hz) > 1e-6 {
			t.Errorf("roundtrip %g Hz -> %g Hz", hz, back)
		}
	}
	if got := slaneyHzToMel(1000); math.Abs(got-15) > 1e-9 {
		t.Errorf("hz_to_mel(1000) = %f, want 15", got)
	}
}
