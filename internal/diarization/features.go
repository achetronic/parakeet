// SPDX-FileCopyrightText: 2026 Alby Hernández <hola@achetronic.com>
// SPDX-License-Identifier: Apache-2.0

package diarization

import "parakeet/internal/mel"

// featureExtractor applies the diarization front-end: pre-emphasis followed by
// the shared log-mel filterbank. The DSP core lives in internal/mel; only the
// pre-emphasis (0.97) is diarization-specific.
type featureExtractor struct {
	cfg Config
	fb  *mel.Filterbank
}

func newFeatureExtractor(cfg Config) *featureExtractor {
	return &featureExtractor{
		cfg: cfg,
		fb:  mel.New(cfg.FeatureSize, cfg.SampleRate, cfg.NFFT, cfg.HopLength, cfg.WinLength),
	}
}

// preemphasis applies y[n] = x[n] - k*x[n-1].
func (e *featureExtractor) preemphasis(samples []float32) []float32 {
	out := make([]float32, len(samples))
	k := float32(e.cfg.Preemphasis)
	for i := 1; i < len(samples); i++ {
		out[i] = samples[i] - k*samples[i-1]
	}
	if len(samples) > 0 {
		out[0] = samples[0]
	}
	return out
}

// extract returns log-mel features [numFrames][featureSize] with pre-emphasis
// and per-feature normalization.
func (e *featureExtractor) extract(samples []float32) [][]float32 {
	return e.fb.Extract(e.preemphasis(samples))
}
