// SPDX-FileCopyrightText: 2026 Alby Hernández <hola@achetronic.com>
// SPDX-License-Identifier: Apache-2.0

package diarization

import (
	"math"
	"testing"
)

func testConfig() Config {
	return Config{
		ChunkLength:                         340,
		ChunkLeftContext:                    0,
		ChunkRightContext:                   40,
		FIFOLength:                          40,
		SpeakerCacheUpdatePeriod:            300,
		SpeakerCacheLength:                  264,
		SpeakerCacheSilenceFramesPerSpeaker: 1,
		PredictionScoreThreshold:            0.25,
		LatestFramesScoreBoost:              0.05,
		MinPositiveScoresRate:               0.5,
		StrongBoostRate:                     0.75,
		WeakBoostRate:                       1.5,
		NumSpeakers:                         8,
		SubsamplingFactor:                   8,
		HiddenSize:                          512,
		FeatureSize:                         128,
		HopLength:                           160,
		NFFT:                                512,
		WinLength:                           400,
		Preemphasis:                         0.97,
		SampleRate:                          16000,
	}
}

func TestNewSpeakerCacheConstants(t *testing.T) {
	c := newSpeakerCache(testConfig())
	if c.minPositiveScores != 16 {
		t.Errorf("minPositiveScores = %d, want 16", c.minPositiveScores)
	}
	if c.numStrongBoostedFrames != 24 {
		t.Errorf("numStrongBoostedFrames = %d, want 24", c.numStrongBoostedFrames)
	}
	if c.numWeakBoostedFrames != 48 {
		t.Errorf("numWeakBoostedFrames = %d, want 48", c.numWeakBoostedFrames)
	}
}

func TestNumPopped(t *testing.T) {
	c := newSpeakerCache(testConfig())
	tests := []struct {
		n    int
		want int
	}{
		{0, 0},
		{40, 0},    // exactly at FIFO capacity
		{41, 41},   // max(300, 1)=300 -> min(300, 41)=41
		{350, 310}, // max(300, 310)=310 -> min(310, 350)=310
	}
	for _, tt := range tests {
		if got := c.numPopped(tt.n); got != tt.want {
			t.Errorf("numPopped(%d) = %d, want %d", tt.n, got, tt.want)
		}
	}
}

func TestUpdateKeepsBounds(t *testing.T) {
	c := newSpeakerCache(testConfig())
	cfg := testConfig()

	silence := make([]float32, cfg.HiddenSize)
	for i := 0; i < 200; i++ {
		numChunk := 40
		lookahead := cfg.ChunkRightContext
		cachedLen := c.numCacheFrames + c.numFifoFrames
		encFrames := cachedLen + numChunk + lookahead

		logits := make([][]float32, encFrames*cfg.SubsamplingFactor)
		for f := range logits {
			row := make([]float32, cfg.NumSpeakers)
			for s := range row {
				row[s] = float32(s%3 - 1) // mix of positive/negative logits
			}
			logits[f] = row
		}
		chunkEmbeds := make([][]float32, numChunk+lookahead)
		for f := range chunkEmbeds {
			row := make([]float32, cfg.HiddenSize)
			for h := range row {
				row[h] = float32(f + h)
			}
			chunkEmbeds[f] = row
		}
		mask := make([]int64, encFrames)
		for m := range mask {
			mask[m] = 1
		}

		c.update(logits, chunkEmbeds, silence, numChunk, mask)

		if c.numCacheFrames > cfg.SpeakerCacheLength {
			t.Fatalf("iteration %d: numCacheFrames=%d exceeds %d", i, c.numCacheFrames, cfg.SpeakerCacheLength)
		}
		if c.numFifoFrames > cfg.FIFOLength {
			t.Fatalf("iteration %d: numFifoFrames=%d exceeds %d", i, c.numFifoFrames, cfg.FIFOLength)
		}
		for f := 0; f < c.numCacheFrames; f++ {
			for _, v := range c.embeds[f] {
				if math.IsNaN(float64(v)) {
					t.Fatalf("iteration %d: NaN in cache embeds", i)
				}
			}
		}
	}
	if !c.isCompressed {
		t.Fatalf("cache never compressed after 200 updates")
	}
}
