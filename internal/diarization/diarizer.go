// SPDX-FileCopyrightText: 2026 Alby Hernández <hola@achetronic.com>
// SPDX-License-Identifier: Apache-2.0

package diarization

import (
	"context"
	"fmt"
	"sort"

	ort "github.com/yalue/onnxruntime_go"

	"parakeet/internal/ortutil"
)

// Segment is a speaker-tagged time span in seconds. Speaker indices follow
// arrival order (0 = first speaker to appear).
type Segment struct {
	Start   float64
	End     float64
	Speaker int
}

// Diarizer runs Nemotron-3-Diarization over 16 kHz mono audio.
type Diarizer struct {
	cfg     Config
	fe      *featureExtractor
	sess    *ort.DynamicAdvancedSession
	envInit bool
}

// New loads the diarization model from modelsDir and creates a Diarizer.
func New(modelsDir string) (*Diarizer, error) {
	cfg, err := loadConfig(modelsDir)
	if err != nil {
		return nil, err
	}

	envInit, err := ortutil.Init()
	if err != nil {
		return nil, err
	}

	modelPath, err := resolveModelPath(modelsDir)
	if err != nil {
		if envInit {
			_ = ort.DestroyEnvironment()
		}
		return nil, err
	}
	sess, err := loadSession(modelPath)
	if err != nil {
		if envInit {
			_ = ort.DestroyEnvironment()
		}
		return nil, err
	}

	return &Diarizer{cfg: cfg, fe: newFeatureExtractor(cfg), sess: sess, envInit: envInit}, nil
}

// Close releases the session and, if this Diarizer initialized the ORT
// environment, destroys it.
func (d *Diarizer) Close() {
	if d.sess != nil {
		d.sess.Destroy()
		d.sess = nil
	}
	if d.envInit {
		_ = ort.DestroyEnvironment()
		d.envInit = false
	}
}

// Diarize returns "who spoke when" as speaker-tagged segments.
func (d *Diarizer) Diarize(ctx context.Context, samples []float32) ([]Segment, error) {
	logits, err := d.run(ctx, samples)
	if err != nil {
		return nil, err
	}
	return extractSegments(logits, d.cfg.NumSpeakers, d.frameDuration()), nil
}

// Activity returns the per-frame speaker probability (sigmoid of the logits)
// at the mel frame rate. The merge uses it to attribute each word to the
// speaker with the highest activity over its span.
func (d *Diarizer) Activity(ctx context.Context, samples []float32) ([][]float32, error) {
	logits, err := d.run(ctx, samples)
	if err != nil {
		return nil, err
	}
	act := make([][]float32, len(logits))
	for f := range logits {
		row := make([]float32, d.cfg.NumSpeakers)
		for s := range row {
			row[s] = sigmoid(logits[f][s])
		}
		act[f] = row
	}
	return act, nil
}

// FrameDuration returns the duration of one activity frame in seconds.
func (d *Diarizer) FrameDuration() float64 { return d.frameDuration() }

func (d *Diarizer) frameDuration() float64 {
	return float64(d.cfg.HopLength) / float64(d.cfg.SampleRate)
}

// run extracts features and runs the chunked encoder, returning the per-frame
// speaker logits (mel rate) for the whole audio.
func (d *Diarizer) run(ctx context.Context, samples []float32) ([][]float32, error) {
	mel := d.fe.extract(samples)
	if len(mel) == 0 {
		return nil, fmt.Errorf("no features extracted")
	}

	sub := d.cfg.SubsamplingFactor
	numMelFrames := len(mel)
	padded := ((numMelFrames + sub - 1) / sub) * sub
	numEncFrames := padded / sub

	cache := newSpeakerCache(d.cfg)
	plans := planChunks(numEncFrames, d.cfg.ChunkLength, d.cfg.ChunkRightContext, sub)

	allLogits := make([][]float32, 0, padded)

	for _, p := range plans {
		if err := ctx.Err(); err != nil {
			return nil, err
		}

		chunkMel := make([][]float32, p.melEnd-p.melStart)
		for f := range chunkMel {
			if src := p.melStart + f; src < numMelFrames {
				chunkMel[f] = mel[src]
			} else {
				chunkMel[f] = make([]float32, d.cfg.FeatureSize)
			}
		}

		cachedEmbeds := cache.getEmbeds()
		cachedLen := len(cachedEmbeds)
		encLen := cachedLen + len(chunkMel)/sub
		mask := make([]int64, encLen)
		for i := range mask {
			mask[i] = 1
		}

		logits, chunkEmbeds, silenceEmbeds, err := d.runChunk(chunkMel, cachedEmbeds, mask)
		if err != nil {
			return nil, err
		}

		cache.update(logits, chunkEmbeds, silenceEmbeds, p.numChunk, mask)

		start := cachedLen * sub
		end := (cachedLen + p.numChunk) * sub
		allLogits = append(allLogits, logits[start:end]...)
	}

	if len(allLogits) > numMelFrames {
		allLogits = allLogits[:numMelFrames]
	}

	return allLogits, nil
}

// runChunk feeds one chunk (mel + cached embeds) to the encoder and returns its
// outputs as owned slices.
func (d *Diarizer) runChunk(mel, cachedEmbeds [][]float32, mask []int64) ([][]float32, [][]float32, []float32, error) {
	chunkMelLen := len(mel)
	cachedLen := len(cachedEmbeds)

	inputData := make([]float32, chunkMelLen*d.cfg.FeatureSize)
	for f := 0; f < chunkMelLen; f++ {
		base := f * d.cfg.FeatureSize
		for m := 0; m < d.cfg.FeatureSize; m++ {
			inputData[base+m] = mel[f][m]
		}
	}

	cacheData := make([]float32, cachedLen*d.cfg.HiddenSize)
	for f := 0; f < cachedLen; f++ {
		base := f * d.cfg.HiddenSize
		for h := 0; h < d.cfg.HiddenSize; h++ {
			cacheData[base+h] = cachedEmbeds[f][h]
		}
	}

	inputTensor, err := ort.NewTensor(ort.NewShape(1, int64(chunkMelLen), int64(d.cfg.FeatureSize)), inputData)
	if err != nil {
		return nil, nil, nil, fmt.Errorf("input_features tensor: %w", err)
	}
	defer inputTensor.Destroy()

	cacheTensor, err := ort.NewTensor(ort.NewShape(1, int64(cachedLen), int64(d.cfg.HiddenSize)), cacheData)
	if err != nil {
		return nil, nil, nil, fmt.Errorf("cached_embeds tensor: %w", err)
	}
	defer cacheTensor.Destroy()

	maskTensor, err := ort.NewTensor(ort.NewShape(1, int64(len(mask))), mask)
	if err != nil {
		return nil, nil, nil, fmt.Errorf("attention_mask tensor: %w", err)
	}
	defer maskTensor.Destroy()

	logitsLen := cachedLen*d.cfg.SubsamplingFactor + chunkMelLen
	chunkEncLen := chunkMelLen / d.cfg.SubsamplingFactor

	logitsT, err := ort.NewEmptyTensor[float32](ort.NewShape(1, int64(logitsLen), int64(d.cfg.NumSpeakers)))
	if err != nil {
		return nil, nil, nil, fmt.Errorf("logits tensor: %w", err)
	}
	defer logitsT.Destroy()

	chunkEmbedsT, err := ort.NewEmptyTensor[float32](ort.NewShape(1, int64(chunkEncLen), int64(d.cfg.HiddenSize)))
	if err != nil {
		return nil, nil, nil, fmt.Errorf("chunk_embeds tensor: %w", err)
	}
	defer chunkEmbedsT.Destroy()

	silenceEmbedsT, err := ort.NewEmptyTensor[float32](ort.NewShape(int64(d.cfg.HiddenSize)))
	if err != nil {
		return nil, nil, nil, fmt.Errorf("silence_embeds tensor: %w", err)
	}
	defer silenceEmbedsT.Destroy()

	if err := d.sess.Run(
		[]ort.Value{inputTensor, cacheTensor, maskTensor},
		[]ort.Value{logitsT, chunkEmbedsT, silenceEmbedsT},
	); err != nil {
		return nil, nil, nil, fmt.Errorf("diarization run: %w", err)
	}

	return extractRows(logitsT.GetData(), logitsLen, d.cfg.NumSpeakers),
		extractRows(chunkEmbedsT.GetData(), chunkEncLen, d.cfg.HiddenSize),
		append([]float32(nil), silenceEmbedsT.GetData()...),
		nil
}

func extractRows(data []float32, rows, cols int) [][]float32 {
	out := make([][]float32, rows)
	for r := 0; r < rows; r++ {
		row := make([]float32, cols)
		copy(row, data[r*cols:(r+1)*cols])
		out[r] = row
	}
	return out
}

// chunkPlan is one offline chunk's window, in mel frames.
type chunkPlan struct {
	melStart int
	melEnd   int
	numChunk int
}

// planChunks lays out offline chunks of chunkLength encoder frames with
// rightContext look-ahead frames.
func planChunks(numEncFrames, chunkLength, rightContext, subsampling int) []chunkPlan {
	var plans []chunkPlan
	for start := 0; start < numEncFrames; start += chunkLength {
		end := start + chunkLength
		if end > numEncFrames {
			end = numEncFrames
		}
		melEnd := end + rightContext
		if melEnd > numEncFrames {
			melEnd = numEncFrames
		}
		plans = append(plans, chunkPlan{
			melStart: start * subsampling,
			melEnd:   melEnd * subsampling,
			numChunk: end - start,
		})
	}
	return plans
}

// extractSegments turns per-frame speaker logits (mel rate) into speech
// segments: sigmoid > 0.5 marks activity, then runs of active frames become
// segments. Overlap yields overlapping segments.
func extractSegments(logits [][]float32, numSpeakers int, frameDuration float64) []Segment {
	numFrames := len(logits)
	if numFrames == 0 {
		return nil
	}
	var segs []Segment
	for s := 0; s < numSpeakers; s++ {
		for f := 0; f < numFrames; {
			if sigmoid(logits[f][s]) <= 0.5 {
				f++
				continue
			}
			start := f
			for f < numFrames && sigmoid(logits[f][s]) > 0.5 {
				f++
			}
			segs = append(segs, Segment{
				Start:   float64(start) * frameDuration,
				End:     float64(f) * frameDuration,
				Speaker: s,
			})
		}
	}
	sort.Slice(segs, func(a, b int) bool {
		if segs[a].Start != segs[b].Start {
			return segs[a].Start < segs[b].Start
		}
		return segs[a].Speaker < segs[b].Speaker
	})
	return segs
}
