// SPDX-FileCopyrightText: 2026 Alby Hernández <hola@achetronic.com>
// SPDX-License-Identifier: Apache-2.0

package diarization

import (
	"math"
	"testing"
)

func TestExtractShape(t *testing.T) {
	cfg := testConfig()
	fe := newFeatureExtractor(cfg)

	samples := make([]float32, 16000) // 1 second of silence
	mel := fe.extract(samples)

	wantFrames := (16000-cfg.WinLength)/cfg.HopLength + 1
	if len(mel) != wantFrames {
		t.Fatalf("got %d frames, want %d", len(mel), wantFrames)
	}
	for f, frame := range mel {
		if len(frame) != cfg.FeatureSize {
			t.Fatalf("frame %d has %d features, want %d", f, len(frame), cfg.FeatureSize)
		}
		for _, v := range frame {
			if math.IsNaN(float64(v)) {
				t.Fatalf("frame %d: NaN in features", f)
			}
		}
	}
}

func TestPreemphasis(t *testing.T) {
	cfg := testConfig()
	fe := newFeatureExtractor(cfg)

	in := make([]float32, 10)
	for i := range in {
		in[i] = 1.0
	}
	out := fe.preemphasis(in)

	if out[0] != 1.0 {
		t.Errorf("out[0] = %v, want 1.0", out[0])
	}
	for i := 1; i < len(out); i++ {
		if math.Abs(float64(out[i]-0.03)) > 1e-6 {
			t.Errorf("out[%d] = %v, want 0.03", i, out[i])
		}
	}
}
