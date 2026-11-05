// SPDX-FileCopyrightText: 2026 Alby Hernández <hola@achetronic.com>
// SPDX-License-Identifier: Apache-2.0

package diarization

import "parakeet/internal/mel"

// featureExtractor applies the diarization front-end: pre-emphasis (0.97)
// followed by the shared log-mel filterbank. The DSP core, including the
// pre-emphasis, lives in internal/mel.
type featureExtractor struct {
	fb *mel.Filterbank
}

func newFeatureExtractor(cfg Config) *featureExtractor {
	return &featureExtractor{
		fb: mel.New(cfg.FeatureSize, cfg.SampleRate, cfg.NFFT, cfg.HopLength, cfg.WinLength, cfg.Preemphasis, false),
	}
}

// extract returns log-mel features [numFrames][featureSize].
func (e *featureExtractor) extract(samples []float32) [][]float32 {
	return e.fb.Extract(samples)
}
