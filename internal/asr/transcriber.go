// SPDX-FileCopyrightText: 2026 Alby Hernández <hola@achetronic.com>
// SPDX-License-Identifier: Apache-2.0

package asr

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"math"
	"os"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"

	ort "github.com/yalue/onnxruntime_go"

	"parakeet/internal/audio"
	"parakeet/internal/mel"
	"parakeet/internal/ortutil"
)

// DebugMode enables verbose logging
var DebugMode bool

// Pre-compiled regex for text cleanup
var whitespaceRegex = regexp.MustCompile(`\s{2,}`)

// Fixed model dimensions for Parakeet TDT 0.6B
const (
	encoderDim         int64 = 1024
	decoderStateDim    int64 = 640
	decoderNumLayers   int64 = 2
	numDurationClasses int64 = 5
)

type Config struct {
	ModelType         string `json:"model_type"`
	FeaturesSize      int    `json:"features_size"`
	SubsamplingFactor int    `json:"subsampling_factor"`
}

// decoderWorker is one concurrent transcription slot: a decoder session with
// reusable tensors, plus, on GPU, its private encoder session so no inference
// session is shared across requests. Each worker has one owner at a time.
type decoderWorker struct {
	// encoder is nil on CPU, where every worker shares Transcriber.encoder.
	encoder   *ort.DynamicAdvancedSession
	session   *ort.AdvancedSession
	encOut    *ort.Tensor[float32]
	targets   *ort.Tensor[int32]
	targetLen *ort.Tensor[int32]
	state1In  *ort.Tensor[float32]
	state2In  *ort.Tensor[float32]
	output    *ort.Tensor[float32]
	state1Out *ort.Tensor[float32]
	state2Out *ort.Tensor[float32]
}

func (w *decoderWorker) destroy() {
	if w.encoder != nil {
		w.encoder.Destroy()
	}
	if w.session != nil {
		w.session.Destroy()
	}
	if w.encOut != nil {
		w.encOut.Destroy()
	}
	if w.targets != nil {
		w.targets.Destroy()
	}
	if w.targetLen != nil {
		w.targetLen.Destroy()
	}
	if w.state1In != nil {
		w.state1In.Destroy()
	}
	if w.state2In != nil {
		w.state2In.Destroy()
	}
	if w.output != nil {
		w.output.Destroy()
	}
	if w.state1Out != nil {
		w.state1Out.Destroy()
	}
	if w.state2Out != nil {
		w.state2Out.Destroy()
	}
}

// newEncoderSession opens the encoder model as a dynamic session so input
// shapes vary with audio length. sessOpts is nil on CPU and device-specific on
// GPU.
func newEncoderSession(encoderPath string, sessOpts *ort.SessionOptions) (*ort.DynamicAdvancedSession, error) {
	return ort.NewDynamicAdvancedSession(
		encoderPath,
		[]string{"audio_signal", "length"},
		[]string{"outputs", "encoded_lengths"},
		sessOpts,
	)
}

func newDecoderWorker(decoderPath string, vocabSize int, sessOpts *ort.SessionOptions) (*decoderWorker, error) {
	w := &decoderWorker{}
	var err error

	outputDim := int64(vocabSize) + numDurationClasses

	w.encOut, err = ort.NewEmptyTensor[float32](ort.NewShape(1, encoderDim, 1))
	if err != nil {
		w.destroy()
		return nil, fmt.Errorf("create encOut tensor: %w", err)
	}

	w.targets, err = ort.NewEmptyTensor[int32](ort.NewShape(1, 1))
	if err != nil {
		w.destroy()
		return nil, fmt.Errorf("create targets tensor: %w", err)
	}

	w.targetLen, err = ort.NewTensor(ort.NewShape(1), []int32{1})
	if err != nil {
		w.destroy()
		return nil, fmt.Errorf("create targetLen tensor: %w", err)
	}

	w.state1In, err = ort.NewEmptyTensor[float32](ort.NewShape(decoderNumLayers, 1, decoderStateDim))
	if err != nil {
		w.destroy()
		return nil, fmt.Errorf("create state1In tensor: %w", err)
	}

	w.state2In, err = ort.NewEmptyTensor[float32](ort.NewShape(decoderNumLayers, 1, decoderStateDim))
	if err != nil {
		w.destroy()
		return nil, fmt.Errorf("create state2In tensor: %w", err)
	}

	w.output, err = ort.NewEmptyTensor[float32](ort.NewShape(1, 1, 1, outputDim))
	if err != nil {
		w.destroy()
		return nil, fmt.Errorf("create output tensor: %w", err)
	}

	w.state1Out, err = ort.NewEmptyTensor[float32](ort.NewShape(decoderNumLayers, 1, decoderStateDim))
	if err != nil {
		w.destroy()
		return nil, fmt.Errorf("create state1Out tensor: %w", err)
	}

	w.state2Out, err = ort.NewEmptyTensor[float32](ort.NewShape(decoderNumLayers, 1, decoderStateDim))
	if err != nil {
		w.destroy()
		return nil, fmt.Errorf("create state2Out tensor: %w", err)
	}

	w.session, err = ort.NewAdvancedSession(
		decoderPath,
		[]string{"encoder_outputs", "targets", "target_length", "input_states_1", "input_states_2"},
		[]string{"outputs", "output_states_1", "output_states_2"},
		[]ort.ArbitraryTensor{w.encOut, w.targets, w.targetLen, w.state1In, w.state2In},
		[]ort.ArbitraryTensor{w.output, w.state1Out, w.state2Out},
		sessOpts,
	)
	if err != nil {
		w.destroy()
		return nil, fmt.Errorf("create decoder session: %w", err)
	}

	return w, nil
}

// Provider selects the ONNX Runtime execution provider used for inference.
type Provider string

const (
	ProviderCPU  Provider = "cpu"
	ProviderCUDA Provider = "cuda"
)

// ParseProvider normalizes a user-supplied provider string. An empty value
// defaults to CPU. Unknown values are rejected so a misconfiguration fails
// loudly at startup instead of silently falling back.
func ParseProvider(s string) (Provider, error) {
	switch Provider(strings.ToLower(strings.TrimSpace(s))) {
	case "", ProviderCPU:
		return ProviderCPU, nil
	case ProviderCUDA:
		return ProviderCUDA, nil
	default:
		return "", fmt.Errorf("unsupported GPU provider %q (supported: cpu, cuda)", s)
	}
}

// GPUConfig selects the execution provider and, for GPU providers, the devices
// to spread workers over. Devices is ignored on CPU.
type GPUConfig struct {
	Provider Provider
	Devices  []int
}

type Transcriber struct {
	config             Config
	vocab              map[int]string
	vocabSize          int
	blankIdx           int
	maxTokensPerStep   int
	chunkFrames        int64
	overlapFrames      int64
	maxEncoderFrames   int64
	longAudio          bool
	disableVADChunking bool
	disableMelChunking bool
	mel                *mel.Filterbank
	encoder            *ort.DynamicAdvancedSession
	vad                *sileroVAD
	decoderPool        chan *decoderWorker
	ffmpeg             *audio.Converter
	envInit            bool
}

// Options groups optional knobs passed to NewTranscriber. Zero values keep
// the previous behavior: WAV-only input, no ffmpeg conversion, CPU inference,
// default chunk sizes, and the full boundary stack (VAD then mel then midpoint).
type Options struct {
	FFmpeg   *audio.Converter
	GPU      GPUConfig
	Chunk    ChunkConfig
	Boundary BoundaryConfig
	// Attention selects the encoder: "full" (default) or "local"
	// (rel_pos_local_attn). Empty means full.
	Attention string
}

// ChunkConfig sets the sliding-window sizes that keep long audio within the
// model's frame limit. Zero values fall back to the package defaults. Enabled
// turns on the windowing; when off, audio over the model limit is rejected.
type ChunkConfig struct {
	Enabled        bool
	Seconds        int
	OverlapSeconds int
}

// BoundaryConfig tunes how the emission boundary inside each chunk overlap is
// chosen. By default the cascade is VAD -> mel energy -> midpoint;
// the disable flags drop the earlier layers so the cascade falls through to the
// next one. VADModelPath points at the Silero VAD ONNX file; when empty the
// caller resolves it to silero_vad.onnx inside the models directory.
type BoundaryConfig struct {
	DisableVAD   bool
	DisableMel   bool
	VADModelPath string
}

// buildSessionOptionsForDevice returns ONNX Runtime session options for one
// device of an execution provider. It returns nil for CPU, so sessions use ORT
// defaults exactly as the pre-GPU path did. For CUDA it pins the session to the
// given device and returns an options object the caller owns and must Destroy
// once every session built from it exists (ORT copies the options at creation
// time).
func buildSessionOptionsForDevice(provider Provider, deviceID int) (*ort.SessionOptions, error) {
	if provider == ProviderCPU || provider == "" {
		return nil, nil
	}

	opts, err := ort.NewSessionOptions()
	if err != nil {
		return nil, fmt.Errorf("create session options: %w", err)
	}

	switch provider {
	case ProviderCUDA:
		cudaOpts, err := ort.NewCUDAProviderOptions()
		if err != nil {
			opts.Destroy()
			return nil, fmt.Errorf("create CUDA provider options: %w", err)
		}
		defer cudaOpts.Destroy()
		if err := cudaOpts.Update(map[string]string{
			"device_id": strconv.Itoa(deviceID),
			// ORT's EXHAUSTIVE default benchmarks every cuDNN algorithm on
			// first use and reserves several GB of workspace, which OOMs the
			// encoder on longer audio. HEURISTIC avoids that up-front cost.
			"cudnn_conv_algo_search": "HEURISTIC",
			// Grow the arena by exactly what is requested; the power-of-two
			// default compounds the cost of the setting above.
			"arena_extend_strategy": "kSameAsRequested",
		}); err != nil {
			opts.Destroy()
			return nil, fmt.Errorf("set CUDA provider options (device %d): %w", deviceID, err)
		}
		if err := opts.AppendExecutionProviderCUDA(cudaOpts); err != nil {
			opts.Destroy()
			return nil, fmt.Errorf("enable CUDA execution provider (device %d): %w", deviceID, err)
		}
	default:
		opts.Destroy()
		return nil, fmt.Errorf("unsupported GPU provider %q (supported: cpu, cuda)", provider)
	}

	return opts, nil
}

// NewTranscriber loads models and initializes the decoder worker pool.
// When opts.FFmpeg is non-nil, non-WAV inputs are transcoded on the fly.
// Otherwise only WAV is accepted and non-WAV inputs return audio.ErrUnsupportedAudio.
func NewTranscriber(modelsDir string, workers int, opts Options) (*Transcriber, error) {
	attention := opts.Attention
	if attention == "" {
		attention = "full"
	}
	t := &Transcriber{
		maxTokensPerStep: 10,
		blankIdx:         8192,
		ffmpeg:           opts.FFmpeg,
		maxEncoderFrames: modelMaxEncoderFrames,
	}
	if attention == "local" {
		t.maxEncoderFrames = localMaxEncoderFrames
	}

	// Load config
	configPath := filepath.Join(modelsDir, "config.json")
	configData, err := os.ReadFile(configPath)
	if err != nil {
		return nil, fmt.Errorf("failed to read config: %w", err)
	}
	if err := json.Unmarshal(configData, &t.config); err != nil {
		return nil, fmt.Errorf("failed to parse config: %w", err)
	}

	if t.config.FeaturesSize == 0 {
		t.config.FeaturesSize = 128
	}
	if t.config.SubsamplingFactor == 0 {
		t.config.SubsamplingFactor = 8
	}

	// Load vocab
	vocabPath := filepath.Join(modelsDir, "vocab.txt")
	if err := t.loadVocab(vocabPath); err != nil {
		return nil, fmt.Errorf("failed to load vocab: %w", err)
	}

	// Initialize mel filterbank
	t.mel = mel.New(t.config.FeaturesSize, 16000, 512, 160, 400, 0.97, false) // NeMo preprocessor: 512 FFT, 10 ms hop, 25 ms window, 0.97 pre-emphasis. Normalisation is applied per window (see transcribeSamples), not over the whole utterance.

	// Resolve chunk sizes (seconds to mel frames) and reject anything that
	// would overrun the model's frame limit.
	chunkSeconds := opts.Chunk.Seconds
	if chunkSeconds <= 0 {
		chunkSeconds = DefaultChunkSeconds
	}
	overlapSeconds := opts.Chunk.OverlapSeconds
	if overlapSeconds < 0 {
		overlapSeconds = DefaultChunkOverlapSeconds
	}
	fps := int64(t.mel.FramesPerSecond())
	t.chunkFrames = int64(chunkSeconds) * fps
	t.overlapFrames = int64(overlapSeconds) * fps
	t.longAudio = opts.Chunk.Enabled
	t.disableVADChunking = opts.Boundary.DisableVAD
	t.disableMelChunking = opts.Boundary.DisableMel
	if t.longAudio {
		if err := validateChunking(t.chunkFrames, t.overlapFrames, int64(t.config.SubsamplingFactor), t.maxEncoderFrames); err != nil {
			return nil, fmt.Errorf("invalid chunk configuration: %w", err)
		}
	}

	// Initialize ONNX Runtime. The first caller in the process initializes the
	// environment and owns its teardown; later callers share it.
	envInit, err := ortutil.Init()
	if err != nil {
		return nil, err
	}
	t.envInit = envInit

	// Resolve encoder path
	encBase := "encoder-model"
	if attention == "local" {
		encBase = "encoder-local"
	}
	encoderPath := filepath.Join(modelsDir, encBase+".int8.onnx")
	if _, err := os.Stat(encoderPath); os.IsNotExist(err) {
		encoderPath = filepath.Join(modelsDir, encBase+".onnx")
		if _, err := os.Stat(encoderPath); os.IsNotExist(err) {
			return nil, fmt.Errorf("encoder model not found (%s); run 'make model-download-asr-int8' or 'make model-download-asr-fp32'", encBase)
		}
	}

	// Resolve decoder path
	decoderPath := filepath.Join(modelsDir, "decoder_joint-model.int8.onnx")
	if _, err := os.Stat(decoderPath); os.IsNotExist(err) {
		decoderPath = filepath.Join(modelsDir, "decoder_joint-model.onnx")
		if _, err := os.Stat(decoderPath); os.IsNotExist(err) {
			return nil, fmt.Errorf("decoder model not found. Download from https://huggingface.co/istupakov/parakeet-tdt-0.6b-v3-onnx")
		}
	}

	// The worker pool is the single concurrency gate: each worker is acquired
	// when a request enters the inference pipeline and held through mel, VAD,
	// encoder, and decoder. On CPU every worker shares one encoder, because a
	// session is safe to run concurrently and host RAM is expensive. On GPU
	// each worker owns its own encoder+decoder pair on one device, so nothing
	// is shared and nothing races; workers are spread round-robin.
	if workers < 1 {
		workers = 1
	}
	isGPU := opts.GPU.Provider == ProviderCUDA
	devices := opts.GPU.Devices
	if isGPU && len(devices) == 0 {
		devices = []int{0}
	}

	// On CPU build the one shared encoder. On GPU the per-worker encoders are
	// created with the workers below.
	if !isGPU {
		t.encoder, err = newEncoderSession(encoderPath, nil)
		if err != nil {
			t.Close()
			return nil, fmt.Errorf("failed to create encoder session: %w", err)
		}
	}

	t.decoderPool = make(chan *decoderWorker, workers)
	for i := 0; i < workers; i++ {
		var sessOpts *ort.SessionOptions
		var enc *ort.DynamicAdvancedSession

		if isGPU {
			deviceID := devices[i%len(devices)]
			sessOpts, err = buildSessionOptionsForDevice(ProviderCUDA, deviceID)
			if err != nil {
				t.Close()
				return nil, fmt.Errorf("configure CUDA device %d: %w", deviceID, err)
			}
			enc, err = newEncoderSession(encoderPath, sessOpts)
			if err != nil {
				sessOpts.Destroy()
				t.Close()
				return nil, fmt.Errorf("create encoder session for worker %d (device %d): %w", i, deviceID, err)
			}
		}

		w, err := newDecoderWorker(decoderPath, t.vocabSize, sessOpts)
		if sessOpts != nil {
			sessOpts.Destroy()
		}
		if err != nil {
			if enc != nil {
				enc.Destroy()
			}
			t.Close()
			return nil, fmt.Errorf("failed to create decoder worker %d: %w", i, err)
		}
		w.encoder = enc
		t.decoderPool <- w
	}

	// Load the Silero VAD model for chunk-boundary selection. It is only useful
	// when long-audio windowing is on, and only when the VAD layer is enabled.
	// A missing model file is not fatal: warn once and let the boundary stack
	// fall back to mel energy. Any other load error is fatal so a
	// corrupt model surfaces loudly at startup.
	if t.longAudio && !t.disableVADChunking {
		vadPath := opts.Boundary.VADModelPath
		if vadPath == "" {
			vadPath = filepath.Join(modelsDir, "silero_vad.onnx")
		}
		// VAD always runs on CPU: on GPU it would share a session outside the
		// pool and race the encoders, and its 32 ms windows gain nothing from
		// GPU kernels.
		vad, err := newSileroVAD(vadPath, nil)
		switch {
		case err == nil:
			t.vad = vad
		case os.IsNotExist(err):
			slog.Warn("VAD model not found, chunk boundaries fall back to mel energy",
				"path", vadPath)
		default:
			t.Close()
			return nil, fmt.Errorf("failed to load Silero VAD model: %w", err)
		}
	}

	slog.Info("transcriber initialized",
		"workers", workers,
		"provider", string(provider(opts.GPU)),
		"encoder", filepath.Base(encoderPath),
		"decoder", filepath.Base(decoderPath),
		"vocabSize", t.vocabSize,
		"vad", t.vad != nil,
	)

	return t, nil
}

// provider returns the effective provider, defaulting empty to CPU, for logging.
func provider(gpu GPUConfig) Provider {
	if gpu.Provider == "" {
		return ProviderCPU
	}
	return gpu.Provider
}

func (t *Transcriber) loadVocab(path string) error {
	file, err := os.Open(path)
	if err != nil {
		return err
	}
	defer file.Close()

	t.vocab = make(map[int]string)
	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		line := scanner.Text()
		parts := strings.SplitN(line, " ", 2)
		if len(parts) != 2 {
			continue
		}
		token := parts[0]
		id, err := strconv.Atoi(parts[1])
		if err != nil {
			continue
		}
		token = strings.ReplaceAll(token, "▁", " ")
		t.vocab[id] = token
		if token == "<blk>" {
			t.blankIdx = id
		}
	}
	t.vocabSize = len(t.vocab)

	if DebugMode {
		slog.Debug("vocab loaded", "tokens", t.vocabSize, "blankIdx", t.blankIdx)
	}

	return scanner.Err()
}

// Close releases the encoder session, all pool workers, and the ONNX Runtime
// environment. Safe to call after requests have run.
func (t *Transcriber) Close() {
	if t.encoder != nil {
		t.encoder.Destroy()
		t.encoder = nil
	}
	if t.vad != nil {
		t.vad.destroy()
		t.vad = nil
	}
	if t.decoderPool != nil {
		close(t.decoderPool)
		for w := range t.decoderPool {
			w.destroy()
		}
	}
	if t.envInit {
		ort.DestroyEnvironment()
	}
}

// Result is one transcription: the full text, the audio length measured from the
// decoded waveform, and timed segments derived from the decoder's token
// positions. Duration is in seconds.
type Result struct {
	Text     string
	Duration float64
	Segments []Segment
}

func (t *Transcriber) Transcribe(ctx context.Context, audioData []byte, format, language string) (string, error) {
	tokens, _, err := t.transcribe(ctx, audioData, format, language, nil)
	if err != nil {
		return "", err
	}
	return t.tokensToText(tokens), nil
}

// TranscribeDetailed is Transcribe plus the measured duration and the timed
// segments the decoder's token positions imply. It runs the identical pipeline;
// the only difference is that it returns the whole Result instead of the text.
func (t *Transcriber) TranscribeDetailed(ctx context.Context, audioData []byte, format, language string) (Result, error) {
	tokens, duration, err := t.transcribe(ctx, audioData, format, language, nil)
	if err != nil {
		return Result{Duration: duration}, err
	}
	return Result{
		Text:     t.tokensToText(tokens),
		Duration: duration,
		Segments: buildSegments(tokens, t.tokenText, t.encoderFrameSeconds(), duration),
	}, nil
}

// TranscribeStream behaves like Transcribe but invokes emit with each new
// chunk of decoded text as soon as the underlying TDT decoder produces it.
// Concatenating all emitted deltas reproduces the transcript verbatim, before
// the final whitespace normalization. The returned full transcript (also sent
// as transcript.text.done) is that same text with leading/trailing whitespace
// trimmed and runs of spaces collapsed, so it may differ from the raw delta
// concatenation by surrounding/duplicate spaces only.
// emit is always called from the same goroutine that called TranscribeStream.
func (t *Transcriber) TranscribeStream(ctx context.Context, audioData []byte, format, language string, emit func(delta string)) (string, error) {
	tokens, _, err := t.transcribe(ctx, audioData, format, language, emit)
	if err != nil {
		return "", err
	}
	return t.tokensToText(tokens), nil
}

// transcribe is the shared implementation. When emit is non-nil, decoded text
// is streamed delta by delta as tokens are produced.
func (t *Transcriber) transcribe(ctx context.Context, audioData []byte, format, language string, emit func(delta string)) ([]decodedToken, float64, error) {
	waveform, err := audio.Load(audioData, t.ffmpeg)
	if err != nil {
		return nil, 0, fmt.Errorf("failed to load audio: %w", err)
	}
	return t.transcribeSamples(ctx, waveform, emit)
}

// acquireWorker waits for a free worker, or returns ctx.Err() when the request
// is canceled while every worker is busy. The caller must return the worker
// with releaseWorker.
func (t *Transcriber) acquireWorker(ctx context.Context) (*decoderWorker, error) {
	select {
	case w := <-t.decoderPool:
		return w, nil
	case <-ctx.Done():
		return nil, ctx.Err()
	}
}

// releaseWorker returns a worker to the pool. The recover guards a send on a
// pool already closed by Close during shutdown.
func (t *Transcriber) releaseWorker(w *decoderWorker) {
	defer func() { _ = recover() }()
	t.decoderPool <- w
}

// transcribeSamples transcribes already-loaded mono 16 kHz samples; it is the
// shared core behind the byte-based and samples-based entry points.
func (t *Transcriber) transcribeSamples(ctx context.Context, waveform []float32, emit func(delta string)) ([]decodedToken, float64, error) {
	select {
	case <-ctx.Done():
		return nil, 0, ctx.Err()
	default:
	}

	// Duration is measured from the decoded waveform, which is already resampled
	// to the mel front end's rate. Deriving it from the uploaded byte count
	// instead would be wrong for every compressed input: an MP3's size says
	// nothing about how long it plays.
	duration := float64(len(waveform)) / float64(t.mel.SampleRate())
	if DebugMode {
		slog.Debug("waveform loaded", "samples", len(waveform), "seconds", duration)
	}

	if len(waveform) < 1600 {
		if DebugMode {
			slog.Debug("audio too short, skipping", "samples", len(waveform))
		}
		return nil, duration, nil
	}

	// Hold one worker for the whole pipeline so at most -workers transcriptions
	// are in flight, bounding mel, VAD, encoder, and decoder together.
	w, err := t.acquireWorker(ctx)
	if err != nil {
		return nil, duration, err
	}
	defer t.releaseWorker(w)

	features := t.mel.Extract(waveform)
	if len(features) == 0 {
		return nil, duration, fmt.Errorf("no features extracted")
	}

	if DebugMode {
		slog.Debug("mel features extracted", "frames", len(features), "featuresPerFrame", len(features[0]))
	}

	subsampling := int64(t.config.SubsamplingFactor)
	// Build the boundary oracle cascade (VAD -> mel energy -> midpoint) over this
	// request's data and plan the chunk windows with it. When long-audio is off
	// the oracle is unused (single window or ErrAudioTooLong).
	oracle := t.newBoundaryOracle(features, waveform)
	plan, err := planForAudioWithBoundaries(int64(len(features)), t.chunkFrames, t.overlapFrames, subsampling, t.maxEncoderFrames, t.longAudio, oracle)
	if err != nil {
		slog.Warn("audio exceeds the single-pass model limit; enable --long-audio to transcribe long files in overlapping chunks",
			"seconds", float64(len(features))/float64(t.mel.FramesPerSecond()),
			"limitSeconds", float64(t.maxEncoderFrames*subsampling)/float64(t.mel.FramesPerSecond()))
		return nil, duration, err
	}

	if DebugMode {
		slog.Debug("chunk plan", "windows", len(plan), "melFrames", len(features), "longAudio", t.longAudio)
	}

	// Decode window by window. Adjacent windows share an overlap, so window i+1's
	// first few tokens are held and compared against window i's tail before they
	// are emitted, dropping seam duplicates and letting the earlier (warmed-up)
	// window win text collisions. Held tokens are released in order
	// before the rest of the window streams, so streaming order is preserved.
	var tokens []decodedToken
	var prevTail []decodedToken
	for i, win := range plan {
		// Emit bounds are the window's owned region expressed in the window's
		// local encoder frames, so tdtDecode drops the overlap it does not own.
		emitStart := melToEncoderFrame(win.emitStart-win.start, subsampling)
		emitEnd := melToEncoderFrame(win.emitEnd-win.start, subsampling)
		// frameOffset turns per-window local timesteps into absolute encoder
		// frames so the seam deduper can align tokens across windows.
		frameOffset := melToEncoderFrame(win.start, subsampling)

		holdFirst := 0
		var resolveSeam func(head []decodedToken) []decodedToken
		if i > 0 {
			holdFirst = seamMaxTokens
			tail := prevTail
			resolveSeam = func(head []decodedToken) []decodedToken {
				return dedupSeam(tail, head)
			}
		}

		// Per-window feature normalisation. NeMo normalises per-feature over one
		// utterance, but long audio is chunked into 60s windows; normalising the
		// whole (possibly hours-long) file at once pushes the statistics far from
		// the ~utterance-length distribution the encoder saw in training, which
		// makes it silently drop whole utterances in the middle of a window. Each
		// window is therefore normalised over its own frames only. For short
		// (single-window) audio this is identical to the old global normalisation.
		windowFeatures := make([][]float32, win.end-win.start)
		for j := win.start; j < win.end; j++ {
			windowFeatures[j-win.start] = append([]float32(nil), features[j]...)
		}
		mel.Normalize(windowFeatures)

		windowTokens, err := t.runInference(ctx, w, windowFeatures, emitStart, emitEnd, frameOffset, holdFirst, resolveSeam, emit)
		if err != nil {
			return nil, duration, fmt.Errorf("inference failed: %w", err)
		}
		tokens = append(tokens, windowTokens...)
		prevTail = windowTokens
	}

	if DebugMode {
		slog.Debug("tokens decoded", "count", len(tokens))
	}

	return tokens, duration, nil
}

// encoderFrameSeconds converts one encoder frame to seconds. The mel front end
// yields sampleRate/hopLength frames per second and the Conformer subsamples by
// config.SubsamplingFactor, so a frame covers subsampling*hop/sampleRate seconds:
// 8 * 160 / 16000 = 0.08 s (12.5 frames per second) for the shipped model.
// Returns 0 when the configuration is unusable, which buildSegments treats as
// "no timing available" rather than inventing a rate.
func (t *Transcriber) encoderFrameSeconds() float64 {
	hop := t.mel.HopLength()
	sampleRate := t.mel.SampleRate()
	if hop <= 0 || sampleRate <= 0 || t.config.SubsamplingFactor <= 0 {
		return 0
	}
	return float64(t.config.SubsamplingFactor) * float64(hop) / float64(sampleRate)
}

// newBoundaryOracle builds the per-request chunk-boundary cascade over this
// request's mel features and waveform: Silero VAD first (when enabled and the
// model loaded), then smoothed mel energy (when enabled), then the arithmetic
// midpoint as the always-decides fallback.
func (t *Transcriber) newBoundaryOracle(features [][]float32, waveform []float32) boundaryOracle {
	var oracles []boundaryOracle
	if !t.disableVADChunking && t.vad != nil {
		oracles = append(oracles, &vadBoundaryOracle{
			vad:       t.vad,
			state:     &vadState{},
			waveform:  waveform,
			hopLength: int64(t.mel.HopLength()),
		})
	}
	if !t.disableMelChunking {
		oracles = append(oracles, newMelEnergyBoundaryOracle(features))
	}
	oracles = append(oracles, midpointBoundaryOracle{})
	return chainBoundaryOracle{oracles: oracles}
}

func (t *Transcriber) runInference(ctx context.Context, w *decoderWorker, features [][]float32, emitStart, emitEnd, frameOffset int64, holdFirst int, resolveSeam func(head []decodedToken) []decodedToken, emit func(delta string)) ([]decodedToken, error) {
	batchSize := int64(1)
	numFeatures := int64(t.config.FeaturesSize)
	numFrames := int64(len(features))

	// Flatten features: [frames, features] → [1, features, frames]
	inputData := make([]float32, numFeatures*numFrames)
	for f := int64(0); f < numFrames; f++ {
		for m := int64(0); m < numFeatures && m < int64(len(features[f])); m++ {
			inputData[m*numFrames+f] = features[f][m]
		}
	}

	inputTensor, err := ort.NewTensor(ort.NewShape(batchSize, numFeatures, numFrames), inputData)
	if err != nil {
		return nil, fmt.Errorf("create input tensor: %w", err)
	}
	defer inputTensor.Destroy()

	lengthTensor, err := ort.NewTensor(ort.NewShape(batchSize), []int64{numFrames})
	if err != nil {
		return nil, fmt.Errorf("create length tensor: %w", err)
	}
	defer lengthTensor.Destroy()

	encodedLen := (numFrames-1)/int64(t.config.SubsamplingFactor) + 1

	outputTensor, err := ort.NewEmptyTensor[float32](ort.NewShape(batchSize, encoderDim, encodedLen))
	if err != nil {
		return nil, fmt.Errorf("create output tensor: %w", err)
	}
	defer outputTensor.Destroy()

	outLenTensor, err := ort.NewEmptyTensor[int64](ort.NewShape(batchSize))
	if err != nil {
		return nil, fmt.Errorf("create output length tensor: %w", err)
	}
	defer outLenTensor.Destroy()

	// Run the encoder: the worker's private session on GPU, the shared session
	// on CPU. The shared CPU encoder is safe to run concurrently.
	enc := w.encoder
	if enc == nil {
		enc = t.encoder
	}
	if err := enc.Run(
		[]ort.Value{inputTensor, lengthTensor},
		[]ort.Value{outputTensor, outLenTensor},
	); err != nil {
		return nil, fmt.Errorf("encoder run failed: %w", err)
	}

	encoderOut := outputTensor.GetData()
	actualEncodedLen := outLenTensor.GetData()[0]

	if DebugMode {
		slog.Debug("encoder output", "floats", len(encoderOut), "encodedLen", actualEncodedLen)
	}

	// Decoder tensors (encoderOut) must remain alive during tdtDecode.
	// The defers above fire after tdtDecode returns, so this is safe.
	return t.tdtDecode(ctx, w, encoderOut, actualEncodedLen, emitStart, emitEnd, frameOffset, holdFirst, resolveSeam, emit)
}

// tdtDecode greedily decodes the encoder output for one window. It decodes the
// whole window so the LSTM state and previous-token feedback stay coherent, but
// only collects and streams tokens whose timestep falls in [emitStart, emitEnd);
// this drops the overlap region owned by an adjacent window. Pass emitStart=0
// and emitEnd=encodedLen to keep everything.
//
// Owned tokens are tagged with an absolute encoder-frame timestep (local
// timestep + frameOffset). When holdFirst > 0 the first holdFirst owned tokens
// are buffered and passed to resolveSeam (the seam deduper) before being
// emitted; the survivors are streamed in order, then the rest of the window
// streams as it is decoded. This keeps streaming order correct while buffering
// only a handful of tokens per seam.
func (t *Transcriber) tdtDecode(ctx context.Context, w *decoderWorker, encoderOut []float32, encodedLen, emitStart, emitEnd, frameOffset int64, holdFirst int, resolveSeam func(head []decodedToken) []decodedToken, emit func(delta string)) ([]decodedToken, error) {
	if DebugMode {
		slog.Debug("TDT decode started", "encoderOutLen", len(encoderOut), "encodedLen", encodedLen)
	}

	// Reset LSTM states to zero for this request
	s1 := w.state1In.GetData()
	s2 := w.state2In.GetData()
	for i := range s1 {
		s1[i] = 0
	}
	for i := range s2 {
		s2[i] = 0
	}

	var result []decodedToken
	var head []decodedToken
	resolved := holdFirst <= 0
	timestep := int64(0)
	emittedTokens := 0
	prevToken := t.blankIdx

	// emitText streams one token's printable text, skipping special <...> tokens.
	emitText := func(id int) {
		if emit == nil {
			return
		}
		if text := t.tokenText(id); text != "" {
			emit(text)
		}
	}
	// flushHead resolves the buffered seam head through the deduper and streams
	// the survivors in order, then marks the seam resolved.
	flushHead := func() {
		survivors := head
		if resolveSeam != nil {
			survivors = resolveSeam(head)
		}
		for _, s := range survivors {
			result = append(result, s)
			emitText(s.id)
		}
		head = nil
		resolved = true
	}

	encOutData := w.encOut.GetData()

	for timestep < encodedLen {
		// Write encoder frame into the reusable encOut tensor
		for d := int64(0); d < encoderDim; d++ {
			idx := d*encodedLen + timestep
			if idx < int64(len(encoderOut)) {
				encOutData[d] = encoderOut[idx]
			} else {
				encOutData[d] = 0
			}
		}

		// Update target token (written directly into tensor backing data)
		w.targets.GetData()[0] = int32(prevToken)

		if err := w.session.Run(); err != nil {
			return nil, fmt.Errorf("decoder run failed: %w", err)
		}

		output := w.output.GetData()
		vocabLogits := output[:t.vocabSize]
		durationLogits := output[t.vocabSize:]

		token := argmax(vocabLogits)
		step := argmax(durationLogits)

		if DebugMode && timestep < 5 {
			slog.Debug("decode step",
				"timestep", timestep,
				"token", token,
				"blank", t.blankIdx,
				"step", step,
				"maxLogit", vocabLogits[token],
			)
		}

		if token != t.blankIdx {
			// Update LSTM states for next step
			copy(w.state1In.GetData(), w.state1Out.GetData())
			copy(w.state2In.GetData(), w.state2Out.GetData())
			prevToken = token
			emittedTokens++
			// Collect and stream only tokens this window owns; the rest belong
			// to an adjacent window's overlap and would duplicate speech.
			if timestep >= emitStart && timestep < emitEnd {
				dt := decodedToken{
					id:       token,
					timestep: frameOffset + timestep,
					duration: int64(step),
					logprob:  logProb(vocabLogits, token),
				}
				if resolved {
					result = append(result, dt)
					emitText(dt.id)
				} else {
					// Hold the window's leading tokens for the seam deduper. Once
					// holdFirst are buffered, resolve and start streaming again.
					head = append(head, dt)
					if len(head) >= holdFirst {
						flushHead()
					}
				}
			}
		}

		// Honor cancellation between decode steps so a disconnected client
		// or an expired deadline frees the worker promptly.
		select {
		case <-ctx.Done():
			if !resolved {
				flushHead()
			}
			return result, ctx.Err()
		default:
		}

		if step > 0 {
			timestep += int64(step)
			emittedTokens = 0
		} else if token == t.blankIdx || emittedTokens >= t.maxTokensPerStep {
			timestep++
			emittedTokens = 0
		}
	}

	// The window ended before holdFirst tokens were seen: resolve whatever the
	// seam head holds (possibly empty) so nothing is left buffered.
	if !resolved {
		flushHead()
	}

	return result, nil
}

func argmax(data []float32) int {
	if len(data) == 0 {
		return 0
	}
	maxIdx := 0
	maxVal := data[0]
	for i, v := range data {
		if v > maxVal {
			maxVal = v
			maxIdx = i
		}
	}
	return maxIdx
}

// logProb returns the log probability of index idx under the decoder's output
// distribution over the vocabulary. The TDT head may export raw logits or
// log-probabilities: logsumexp normalises the first case and is ~0 for the
// second, so the result is a log probability either way. It is computed only for
// tokens that are actually emitted, not for every decode step.
func logProb(logits []float32, idx int) float32 {
	if len(logits) == 0 || idx < 0 || idx >= len(logits) {
		return 0
	}
	max := logits[0]
	for _, v := range logits {
		if v > max {
			max = v
		}
	}
	sum := 0.0
	for _, v := range logits {
		sum += math.Exp(float64(v) - float64(max))
	}
	if sum <= 0 {
		return 0
	}
	return float32(float64(logits[idx]) - float64(max) - math.Log(sum))
}

// tokenText returns the printable text for a token id, or "" for unknown tokens
// and special <...> markers. vocab values already have word-boundary marks
// (U+2581) translated to spaces at load time (see loadVocab), so the text is
// returned as-is.
func (t *Transcriber) tokenText(id int) string {
	text, ok := t.vocab[id]
	if !ok {
		return ""
	}
	if strings.HasPrefix(text, "<") && strings.HasSuffix(text, ">") {
		return ""
	}
	return text
}

func (t *Transcriber) tokensToText(tokens []decodedToken) string {
	var parts []string
	for _, tok := range tokens {
		if text := t.tokenText(tok.id); text != "" {
			parts = append(parts, text)
		}
	}
	text := strings.Join(parts, "")
	text = strings.TrimSpace(whitespaceRegex.ReplaceAllString(text, " "))
	return text
}
