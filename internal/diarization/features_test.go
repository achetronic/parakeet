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
	features := fe.extract(samples)

	// Centred STFT: len(samples)/hop frames, matching the shared mel frontend.
	wantFrames := 16000 / cfg.HopLength
	if len(features) != wantFrames {
		t.Fatalf("got %d frames, want %d", len(features), wantFrames)
	}
	for f, frame := range features {
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
