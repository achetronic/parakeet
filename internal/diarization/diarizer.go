// SPDX-FileCopyrightText: 2026 Alby Hernández <hola@achetronic.com>
// SPDX-License-Identifier: Apache-2.0

package diarization

import (
	"context"
	"fmt"
	"log/slog"
	"sort"
	"sync"
	"time"

	ort "github.com/yalue/onnxruntime_go"

	"parakeet/internal/idleunload"
	"parakeet/internal/memtrim"
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
	sem     chan struct{}
	envInit bool

	// Kept so the session can be rebuilt after an unload; the model config and
	// the feature extractor are cheap and stay resident.
	modelPath string

	// mu guards the session. A diarization holds it for reading for its whole
	// run, so Unload (write) only destroys the session once nothing is in flight.
	mu     sync.RWMutex
	loaded bool

	// idle frees the session after a quiet period.
	idle *idleunload.Watcher
}

// New loads the diarization model from modelsDir and creates a Diarizer.
// workers bounds how many diarization runs may execute at once.
func New(modelsDir string, workers int) (*Diarizer, error) {
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

	if workers < 1 {
		workers = 1
	}
	d := &Diarizer{
		cfg:       cfg,
		fe:        newFeatureExtractor(cfg),
		modelPath: modelPath,
		sem:       make(chan struct{}, workers),
		envInit:   envInit,
	}
	d.idle = idleunload.New("diarization", d.unloadIfIdle)

	if err := d.loadSession(); err != nil {
		d.Close()
		return nil, err
	}
	return d, nil
}

// loadSession builds the ONNX session. The caller holds mu for writing.
func (d *Diarizer) loadSession() error {
	sess, err := loadSession(d.modelPath)
	if err != nil {
		return err
	}
	d.sess = sess
	d.loaded = true
	d.idle.Touch()
	slog.Info("diarization model loaded", "model", d.modelPath)
	return nil
}

// unloadSession destroys the ONNX session. The caller holds mu for writing.
func (d *Diarizer) unloadSession(reason string) {
	if d.sess != nil {
		d.sess.Destroy()
		d.sess = nil
	}
	d.loaded = false
	// Freed by ORT, retained by the libc arenas: hand it back to the OS.
	memtrim.Release()
	slog.Info("diarization model unloaded", "reason", reason)
}

// acquireSessions loads the session when needed and holds it for the run. The
// returned release func must be called when the run is done.
func (d *Diarizer) acquireSessions(ctx context.Context) (func(), error) {
	for {
		d.mu.RLock()
		if d.loaded {
			d.idle.Touch()
			return func() {
				d.mu.RUnlock()
				d.idle.Touch()
			}, nil
		}
		d.mu.RUnlock()

		d.mu.Lock()
		if !d.loaded {
			err := ctx.Err()
			if err == nil {
				err = d.loadSession()
			}
			if err != nil {
				d.mu.Unlock()
				return nil, err
			}
		}
		d.mu.Unlock()
		// A concurrent unload can slip in here; the loop just loads again.
	}
}

// Load brings the diarization session into memory. Idempotent.
func (d *Diarizer) Load(ctx context.Context) error {
	release, err := d.acquireSessions(ctx)
	if err != nil {
		return err
	}
	release()
	return nil
}

// Unload releases the diarization session, freeing its memory. In-flight runs
// finish first.
func (d *Diarizer) Unload() {
	d.unload("manual")
}

func (d *Diarizer) unload(reason string) bool {
	d.mu.Lock()
	defer d.mu.Unlock()
	if !d.loaded {
		return false
	}
	d.unloadSession(reason)
	return true
}

// Loaded reports whether the diarization session is in memory.
func (d *Diarizer) Loaded() bool {
	d.mu.RLock()
	defer d.mu.RUnlock()
	return d.loaded
}

// StartIdleUnload frees the session after timeout without diarization requests.
// Zero or a negative timeout disables it.
func (d *Diarizer) StartIdleUnload(timeout time.Duration) {
	d.idle.Start(timeout)
}

// unloadIfIdle frees the session only when it has been idle for the whole
// timeout, re-checking under the write lock so a slow run is not unloaded the
// instant it finishes.
func (d *Diarizer) unloadIfIdle(timeout time.Duration) bool {
	d.mu.Lock()
	defer d.mu.Unlock()
	if !d.loaded || d.idle.IdleFor() < timeout {
		return false
	}
	d.unloadSession("idle timeout")
	return true
}

// Close stops the idle unloader, releases the session and, if this Diarizer
// initialized the ORT environment, destroys it.
func (d *Diarizer) Close() {
	d.idle.Stop()
	d.mu.Lock()
	if d.loaded {
		d.unloadSession("shutdown")
	}
	d.mu.Unlock()
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
	// The session must stay alive for the whole run: hold the read side so a
	// concurrent unload waits instead of pulling it out from under this run.
	releaseSession, err := d.acquireSessions(ctx)
	if err != nil {
		return nil, err
	}
	defer releaseSession()

	// Bound concurrent runs. The session is CPU-only and shared, so the
	// semaphore caps concurrency and memory, not correctness.
	select {
	case d.sem <- struct{}{}:
	case <-ctx.Done():
		return nil, ctx.Err()
	}
	defer func() { <-d.sem }()

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
