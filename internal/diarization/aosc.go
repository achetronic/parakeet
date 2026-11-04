// SPDX-FileCopyrightText: 2026 Alby Hernández <hola@achetronic.com>
// SPDX-License-Identifier: Apache-2.0

package diarization

import (
	"math"
	"sort"
)

// speakerCache is the Arrival-Order Speaker Cache (AOSC) plus FIFO queue of
// Streaming Sortformer. It keeps speaker identity consistent across chunks.
// Ported from Nemotron3DiarizationSpeakerCache in Transformers.
type speakerCache struct {
	cfg Config

	minPositiveScores      int
	numStrongBoostedFrames int
	numWeakBoostedFrames   int

	embeds         [][]float32 // [speakerCacheLength][hidden]
	probs          [][]float32 // [speakerCacheLength][numSpeakers]
	fifo           [][]float32 // [fifoLength][hidden]
	numCacheFrames int
	numFifoFrames  int
	isCompressed   bool
}

func newSpeakerCache(cfg Config) *speakerCache {
	budget := cfg.SpeakerCacheLength/cfg.NumSpeakers - cfg.SpeakerCacheSilenceFramesPerSpeaker
	return &speakerCache{
		cfg:                    cfg,
		minPositiveScores:      int(math.Floor(float64(budget) * cfg.MinPositiveScoresRate)),
		numStrongBoostedFrames: int(math.Floor(float64(budget) * cfg.StrongBoostRate)),
		numWeakBoostedFrames:   int(math.Floor(float64(budget) * cfg.WeakBoostRate)),
		embeds:                 make([][]float32, cfg.SpeakerCacheLength),
		probs:                  make([][]float32, cfg.SpeakerCacheLength),
		fifo:                   make([][]float32, cfg.FIFOLength),
	}
}

// getEmbeds returns the cached frames to prepend to the current chunk:
// concat(cache frames, fifo frames).
func (c *speakerCache) getEmbeds() [][]float32 {
	out := make([][]float32, 0, c.numCacheFrames+c.numFifoFrames)
	out = append(out, c.embeds[:c.numCacheFrames]...)
	out = append(out, c.fifo[:c.numFifoFrames]...)
	return out
}

// poolProbs averages per-frame speaker probabilities (mel rate) down to the
// encoder frame rate (subsampling frames per pool window), zeroing masked
// frames.
func (c *speakerCache) poolProbs(logits [][]float32, mask []int64) [][]float32 {
	factor := c.cfg.SubsamplingFactor
	numEncFrames := len(logits) / factor
	probs := make([][]float32, numEncFrames)
	for f := 0; f < numEncFrames; f++ {
		row := make([]float32, c.cfg.NumSpeakers)
		for s := 0; s < c.cfg.NumSpeakers; s++ {
			var sum float32
			for k := 0; k < factor; k++ {
				sum += sigmoid(logits[f*factor+k][s])
			}
			row[s] = sum / float32(factor)
		}
		probs[f] = row
	}
	if mask != nil {
		for f := 0; f < numEncFrames && f < len(mask); f++ {
			if mask[f] == 0 {
				for s := 0; s < c.cfg.NumSpeakers; s++ {
					probs[f][s] = 0
				}
			}
		}
	}
	return probs
}

// numPopped returns how many FIFO frames move to the speaker cache this step.
func (c *speakerCache) numPopped(numFifoFrames int) int {
	if numFifoFrames <= c.cfg.FIFOLength {
		return 0
	}
	popped := c.cfg.SpeakerCacheUpdatePeriod
	if overflow := numFifoFrames - c.cfg.FIFOLength; overflow > popped {
		popped = overflow
	}
	if popped > numFifoFrames {
		popped = numFifoFrames
	}
	return popped
}

// update pushes a processed chunk onto the FIFO, moving its oldest frames to
// the speaker cache when it overflows. chunkLogits is at the mel rate (full
// cached+chunk output); chunkEmbeds is the chunk's encoder-rate embeddings
// (chunk + look-ahead); numChunkFrames excludes the look-ahead.
func (c *speakerCache) update(chunkLogits, chunkEmbeds [][]float32, silenceEmbeds []float32, numChunkFrames int, mask []int64) {
	probs := c.poolProbs(chunkLogits, mask)

	// Copy the chunk's own frames (excluding look-ahead) so the cache owns its
	// data after the ONNX output tensors are destroyed.
	chunkFrames := make([][]float32, numChunkFrames)
	for i := 0; i < numChunkFrames && i < len(chunkEmbeds); i++ {
		chunkFrames[i] = append([]float32(nil), chunkEmbeds[i]...)
	}

	newFifo := make([][]float32, 0, c.numFifoFrames+numChunkFrames)
	newFifo = append(newFifo, c.fifo[:c.numFifoFrames]...)
	newFifo = append(newFifo, chunkFrames...)

	popped := c.numPopped(len(newFifo))
	if popped > 0 {
		chunkStart := c.numCacheFrames + c.numFifoFrames
		fifoProbs := make([][]float32, 0, c.numFifoFrames+numChunkFrames)
		fifoProbs = append(fifoProbs, probs[c.numCacheFrames:c.numCacheFrames+c.numFifoFrames]...)
		fifoProbs = append(fifoProbs, probs[chunkStart:chunkStart+numChunkFrames]...)

		var storedProbs [][]float32
		if c.isCompressed {
			storedProbs = c.probs[:c.numCacheFrames]
		} else {
			storedProbs = probs[:c.numCacheFrames]
		}

		cacheEmbeds := make([][]float32, 0, c.numCacheFrames+popped)
		cacheEmbeds = append(cacheEmbeds, c.embeds[:c.numCacheFrames]...)
		cacheEmbeds = append(cacheEmbeds, newFifo[:popped]...)
		cacheProbs := make([][]float32, 0, c.numCacheFrames+popped)
		cacheProbs = append(cacheProbs, storedProbs...)
		cacheProbs = append(cacheProbs, fifoProbs[:popped]...)

		newFifo = newFifo[popped:]

		if len(cacheEmbeds) > c.cfg.SpeakerCacheLength {
			cacheEmbeds, cacheProbs = c.compress(cacheEmbeds, cacheProbs, silenceEmbeds)
			c.isCompressed = true
		}
		c.numCacheFrames = len(cacheEmbeds)
		copy(c.embeds, cacheEmbeds)
		copy(c.probs, cacheProbs)
	}

	c.numFifoFrames = len(newFifo)
	copy(c.fifo, newFifo)
}

// getFrameScores scores each (frame, speaker) by its log-odds of activity.
func (c *speakerCache) getFrameScores(probs [][]float32) [][]float32 {
	threshold := float32(c.cfg.PredictionScoreThreshold)
	numFrames := len(probs)
	numSpeakers := c.cfg.NumSpeakers
	scores := make([][]float32, numFrames)

	for f := 0; f < numFrames; f++ {
		var sumLogComp float32
		for s := 0; s < numSpeakers; s++ {
			lc := 1 - probs[f][s]
			if lc < threshold {
				lc = threshold
			}
			sumLogComp += float32(math.Log(float64(lc)))
		}
		row := make([]float32, numSpeakers)
		for s := 0; s < numSpeakers; s++ {
			lp := probs[f][s]
			if lp < threshold {
				lp = threshold
			}
			lc := 1 - probs[f][s]
			if lc < threshold {
				lc = threshold
			}
			row[s] = float32(math.Log(float64(lp))) - float32(math.Log(float64(lc))) + sumLogComp - float32(math.Log(0.5))
		}
		scores[f] = row
	}

	isSpeech := make([][]bool, numFrames)
	for f := 0; f < numFrames; f++ {
		isSpeech[f] = make([]bool, numSpeakers)
		for s := 0; s < numSpeakers; s++ {
			isSpeech[f][s] = probs[f][s] > 0.5
			if !isSpeech[f][s] {
				scores[f][s] = float32(math.Inf(-1))
			}
		}
	}

	hasEnough := make([]bool, numSpeakers)
	for s := 0; s < numSpeakers; s++ {
		count := 0
		for f := 0; f < numFrames; f++ {
			if scores[f][s] > 0 {
				count++
			}
		}
		hasEnough[s] = count >= c.minPositiveScores
	}
	for f := 0; f < numFrames; f++ {
		for s := 0; s < numSpeakers; s++ {
			if scores[f][s] <= 0 && isSpeech[f][s] && hasEnough[s] {
				scores[f][s] = float32(math.Inf(-1))
			}
		}
	}
	return scores
}

// boostScores adds boost to the numBoosted highest-scoring frames of each
// speaker.
func (c *speakerCache) boostScores(scores [][]float32, numBoosted int, boost float32) {
	numFrames := len(scores)
	if numFrames == 0 {
		return
	}
	numSpeakers := c.cfg.NumSpeakers
	idx := make([]int, numFrames)
	for s := 0; s < numSpeakers; s++ {
		for f := 0; f < numFrames; f++ {
			idx[f] = f
		}
		sort.Slice(idx, func(a, b int) bool { return scores[idx[a]][s] > scores[idx[b]][s] })
		for k := 0; k < numBoosted && k < numFrames; k++ {
			scores[idx[k]][s] += boost
		}
	}
}

// compress keeps the speakerCacheLength most important frames, grouped by
// speaker, filling reserved slots with the silence embedding.
func (c *speakerCache) compress(embeds, probs [][]float32, silenceEmbeds []float32) ([][]float32, [][]float32) {
	numFrames := len(probs)
	numSpeakers := c.cfg.NumSpeakers
	scores := c.getFrameScores(probs)

	if numFrames > c.cfg.SpeakerCacheLength {
		boost := float32(c.cfg.LatestFramesScoreBoost)
		for f := c.cfg.SpeakerCacheLength; f < numFrames; f++ {
			for s := 0; s < numSpeakers; s++ {
				scores[f][s] += boost
			}
		}
	}
	c.boostScores(scores, c.numStrongBoostedFrames, -2*float32(math.Log(0.5)))
	c.boostScores(scores, c.numWeakBoostedFrames, -float32(math.Log(0.5)))

	numScoredFrames := numFrames + c.cfg.SpeakerCacheSilenceFramesPerSpeaker
	sentinel := numScoredFrames * numSpeakers

	type scored struct {
		score float32
		idx   int
	}
	flat := make([]scored, 0, numScoredFrames*numSpeakers)
	for s := 0; s < numSpeakers; s++ {
		for f := 0; f < numFrames; f++ {
			flat = append(flat, scored{scores[f][s], s*numScoredFrames + f})
		}
		flat = append(flat, scored{float32(math.Inf(1)), s*numScoredFrames + numFrames})
	}
	sort.Slice(flat, func(a, b int) bool { return flat[a].score > flat[b].score })

	topIdx := make([]int, c.cfg.SpeakerCacheLength)
	for i := 0; i < c.cfg.SpeakerCacheLength; i++ {
		if flat[i].score == float32(math.Inf(-1)) {
			topIdx[i] = sentinel
		} else {
			topIdx[i] = flat[i].idx
		}
	}
	sort.Ints(topIdx)

	outEmbeds := make([][]float32, c.cfg.SpeakerCacheLength)
	outProbs := make([][]float32, c.cfg.SpeakerCacheLength)
	for i, idx := range topIdx {
		if idx == sentinel {
			outEmbeds[i] = append([]float32(nil), silenceEmbeds...)
			outProbs[i] = make([]float32, numSpeakers)
			continue
		}
		fi := idx % numScoredFrames
		if fi >= numFrames {
			outEmbeds[i] = append([]float32(nil), silenceEmbeds...)
			outProbs[i] = make([]float32, numSpeakers)
			continue
		}
		outEmbeds[i] = append([]float32(nil), embeds[fi]...)
		outProbs[i] = append([]float32(nil), probs[fi]...)
	}
	return outEmbeds, outProbs
}

func sigmoid(x float32) float32 {
	return float32(1.0 / (1.0 + math.Exp(float64(-x))))
}
