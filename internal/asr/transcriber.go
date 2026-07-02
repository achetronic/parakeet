// SPDX-FileCopyrightText: 2026 Alby Hernández <hola@achetronic.com>
// SPDX-License-Identifier: Apache-2.0

package asr

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"os"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"

	ort "github.com/yalue/onnxruntime_go"
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

// decoderWorker holds a pre-initialized decoder session with reusable tensors.
// Each worker is owned by at most one goroutine at a time via the pool channel.
type decoderWorker struct {
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

// GPUConfig selects the execution provider and, for GPU providers, the device.
type GPUConfig struct {
	Provider Provider
	DeviceID int
}

type Transcriber struct {
	config             Config
	vocab              map[int]string
	vocabSize          int
	blankIdx           int
	maxTokensPerStep   int
	chunkFrames        int64
	overlapFrames      int64
	longAudio          bool
	disableVADChunking bool
	disableMelChunking bool
	mel                *MelFilterbank
	encoder            *ort.DynamicAdvancedSession
	vad                *sileroVAD
	decoderPool        chan *decoderWorker
	ffmpeg             *ffmpegConverter
}

// Options groups optional knobs passed to NewTranscriber. Zero values keep
// the previous behavior: WAV-only input, no ffmpeg conversion, CPU inference,
// default chunk sizes, and the full boundary stack (VAD then mel then midpoint).
type Options struct {
	FFmpeg   FFmpegConfig
	GPU      GPUConfig
	Chunk    ChunkConfig
	Boundary BoundaryConfig
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

// buildSessionOptions returns the ONNX Runtime session options for the
// configured execution provider. It returns (nil, nil) for the CPU provider so
// sessions are created with default CPU behavior, identical to the pre-GPU code
// path. For a GPU provider it returns a configured *ort.SessionOptions that the
// caller owns and must Destroy after all sessions are created (ORT copies the
// options into each session at creation time, so the object is safe to free
// once sessions exist). A future execution provider is added here.
func buildSessionOptions(gpu GPUConfig) (*ort.SessionOptions, error) {
	if gpu.Provider == ProviderCPU || gpu.Provider == "" {
		return nil, nil
	}

	opts, err := ort.NewSessionOptions()
	if err != nil {
		return nil, fmt.Errorf("create session options: %w", err)
	}

	switch gpu.Provider {
	case ProviderCUDA:
		cudaOpts, err := ort.NewCUDAProviderOptions()
		if err != nil {
			opts.Destroy()
			return nil, fmt.Errorf("create CUDA provider options: %w", err)
		}
		defer cudaOpts.Destroy()
		if err := cudaOpts.Update(map[string]string{
			"device_id": strconv.Itoa(gpu.DeviceID),
			// EXHAUSTIVE (the ORT default) benchmarks every cuDNN convolution
			// algorithm on first run and can reserve many GB of workspace for a
			// single Conv, which OOMs the encoder on longer audio. HEURISTIC
			// picks a good algorithm without that up-front allocation.
			"cudnn_conv_algo_search": "HEURISTIC",
			// Grow the GPU arena by exactly what is requested instead of the
			// default power-of-two steps, which otherwise compounds the above.
			"arena_extend_strategy": "kSameAsRequested",
		}); err != nil {
			opts.Destroy()
			return nil, fmt.Errorf("set CUDA provider options (device %d): %w", gpu.DeviceID, err)
		}
		if err := opts.AppendExecutionProviderCUDA(cudaOpts); err != nil {
			opts.Destroy()
			return nil, fmt.Errorf("enable CUDA execution provider (device %d): %w", gpu.DeviceID, err)
		}
	default:
		opts.Destroy()
		return nil, fmt.Errorf("unsupported GPU provider %q (supported: cpu, cuda)", gpu.Provider)
	}

	return opts, nil
}

// NewTranscriber loads models and initializes the decoder worker pool.
// When opts.FFmpeg.Enabled is true and the ffmpeg binary is resolvable,
// non-WAV inputs will be transcoded on the fly. Otherwise, only WAV is
// accepted and non-WAV inputs return ErrUnsupportedAudio.
func NewTranscriber(modelsDir string, workers int, opts Options) (*Transcriber, error) {
	t := &Transcriber{
		maxTokensPerStep: 10,
		blankIdx:         8192,
		ffmpeg:           newFFmpegConverter(opts.FFmpeg),
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
	t.mel = NewMelFilterbank(t.config.FeaturesSize, 16000)

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
		if err := validateChunking(t.chunkFrames, t.overlapFrames, int64(t.config.SubsamplingFactor)); err != nil {
			return nil, fmt.Errorf("invalid chunk configuration: %w", err)
		}
	}

	// Initialize ONNX Runtime
	libPath := os.Getenv("ONNXRUNTIME_LIB")
	if libPath == "" {
		commonPaths := []string{
			"/usr/lib/libonnxruntime.so",
			"/usr/lib/x86_64-linux-gnu/libonnxruntime.so",
			"/usr/local/lib/libonnxruntime.so",
			"/opt/onnxruntime/lib/libonnxruntime.so",
			"./libonnxruntime.so",
			"libonnxruntime.so.1.25.1",
		}
		for _, p := range commonPaths {
			if _, err := os.Stat(p); err == nil {
				libPath = p
				break
			}
		}
	}
	if libPath == "" {
		return nil, fmt.Errorf("ONNX Runtime library not found. Set ONNXRUNTIME_LIB env var or install libonnxruntime")
	}

	ort.SetSharedLibraryPath(libPath)
	if err := ort.InitializeEnvironment(); err != nil {
		return nil, fmt.Errorf("failed to initialize ONNX Runtime: %w", err)
	}

	// Resolve encoder path
	encoderPath := filepath.Join(modelsDir, "encoder-model.int8.onnx")
	if _, err := os.Stat(encoderPath); os.IsNotExist(err) {
		encoderPath = filepath.Join(modelsDir, "encoder-model.onnx")
		if _, err := os.Stat(encoderPath); os.IsNotExist(err) {
			return nil, fmt.Errorf("encoder model not found. Download from https://huggingface.co/istupakov/parakeet-tdt-0.6b-v3-onnx")
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

	// Build execution-provider session options. nil for CPU (default behavior);
	// a configured object for GPU that we own and destroy once every session
	// below has been created (ORT copies options into each session).
	sessOpts, err := buildSessionOptions(opts.GPU)
	if err != nil {
		return nil, fmt.Errorf("failed to configure execution provider: %w", err)
	}
	if sessOpts != nil {
		defer sessOpts.Destroy()
	}

	// Encoder runs as a single long-lived dynamic session reused across requests.
	// Input/output shapes vary with audio length, so we pass freshly shaped
	// tensors to each Run rather than rebuilding the session. ORT Run is
	// thread-safe on a shared session and every request supplies its own
	// tensors, so this is safe under the concurrent decoder worker model.
	t.encoder, err = ort.NewDynamicAdvancedSession(
		encoderPath,
		[]string{"audio_signal", "length"},
		[]string{"outputs", "encoded_lengths"},
		sessOpts,
	)
	if err != nil {
		return nil, fmt.Errorf("failed to create encoder session: %w", err)
	}

	// Create decoder worker pool — each worker owns a persistent session and
	// pre-allocated tensors. Workers are acquired per request and returned after.
	if workers < 1 {
		workers = 1
	}
	t.decoderPool = make(chan *decoderWorker, workers)
	for i := 0; i < workers; i++ {
		w, err := newDecoderWorker(decoderPath, t.vocabSize, sessOpts)
		if err != nil {
			t.Close()
			return nil, fmt.Errorf("failed to create decoder worker %d: %w", i, err)
		}
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
		vad, err := newSileroVAD(vadPath, sessOpts)
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
	ort.DestroyEnvironment()
}

func (t *Transcriber) Transcribe(ctx context.Context, audioData []byte, format, language string) (string, error) {
	return t.transcribe(ctx, audioData, format, language, nil)
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
	return t.transcribe(ctx, audioData, format, language, emit)
}

// transcribe is the shared implementation. When emit is non-nil, decoded text
// is streamed delta by delta as tokens are produced.
func (t *Transcriber) transcribe(ctx context.Context, audioData []byte, format, language string, emit func(delta string)) (string, error) {
	// Let's check context immediately
	select {
	case <-ctx.Done():
		return "", ctx.Err()
	default:
	}

	waveform, err := t.loadAudio(audioData, format)
	if err != nil {
		return "", fmt.Errorf("failed to load audio: %w", err)
	}

	if DebugMode {
		slog.Debug("waveform loaded", "samples", len(waveform), "seconds", float64(len(waveform))/16000.0)
	}

	if len(waveform) < 1600 {
		if DebugMode {
			slog.Debug("audio too short, skipping", "samples", len(waveform))
		}
		return "", nil
	}

	features := t.mel.Extract(waveform)
	if len(features) == 0 {
		return "", fmt.Errorf("no features extracted")
	}

	if DebugMode {
		slog.Debug("mel features extracted", "frames", len(features), "featuresPerFrame", len(features[0]))
	}

	subsampling := int64(t.config.SubsamplingFactor)
	// Build the boundary oracle cascade (VAD -> mel energy -> midpoint) over this
	// request's data and plan the chunk windows with it. When long-audio is off
	// the oracle is unused (single window or ErrAudioTooLong).
	oracle := t.newBoundaryOracle(features, waveform)
	plan, err := planForAudioWithBoundaries(int64(len(features)), t.chunkFrames, t.overlapFrames, subsampling, t.longAudio, oracle)
	if err != nil {
		slog.Warn("audio exceeds the single-pass model limit; enable --long-audio to transcribe long files in overlapping chunks",
			"seconds", float64(len(features))/float64(t.mel.FramesPerSecond()),
			"limitSeconds", float64(modelMaxEncoderFrames*subsampling)/float64(t.mel.FramesPerSecond()))
		return "", err
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

		windowTokens, err := t.runInference(ctx, features[win.start:win.end], emitStart, emitEnd, frameOffset, holdFirst, resolveSeam, emit)
		if err != nil {
			return "", fmt.Errorf("inference failed: %w", err)
		}
		tokens = append(tokens, windowTokens...)
		prevTail = windowTokens
	}

	if DebugMode {
		slog.Debug("tokens decoded", "count", len(tokens))
	}

	return t.tokensToText(tokens), nil
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

// loadAudio decodes raw request bytes into mono 16 kHz float32 samples.
//
// Detection is done by content, not by filename extension: an OpenAI client
// is free to upload a file without an extension or with a misleading one,
// and the transcription endpoint only ever sees bytes. WAV inputs are
// parsed in-process with zero external dependencies. Anything else is
// delegated to the optional ffmpeg converter; when ffmpeg is unavailable
// the call fails with ErrUnsupportedAudio so the HTTP layer can surface a
// 400 response instead of a generic 500.
//
// The `format` parameter is kept for logging and future heuristics, but it
// is intentionally not used to pick the decoder.
func (t *Transcriber) loadAudio(data []byte, format string) ([]float32, error) {
	if isWAV(data) {
		return parseWAV(data)
	}

	if t.ffmpeg == nil {
		return nil, fmt.Errorf("input is not WAV and ffmpeg conversion is disabled: %w", ErrUnsupportedAudio)
	}

	if DebugMode {
		slog.Debug("converting audio via ffmpeg",
			"format", format,
			"bytes", len(data),
		)
	}

	wavData, err := t.ffmpeg.Convert(data)
	if err != nil {
		return nil, err
	}
	return parseWAV(wavData)
}

func (t *Transcriber) runInference(ctx context.Context, features [][]float32, emitStart, emitEnd, frameOffset int64, holdFirst int, resolveSeam func(head []decodedToken) []decodedToken, emit func(delta string)) ([]decodedToken, error) {
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

	// Reuse the shared encoder session. Shapes vary per request, so tensors are
	// supplied to Run each time; the session itself is built once at startup.
	if err := t.encoder.Run(
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
	return t.tdtDecode(ctx, encoderOut, actualEncodedLen, emitStart, emitEnd, frameOffset, holdFirst, resolveSeam, emit)
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
func (t *Transcriber) tdtDecode(ctx context.Context, encoderOut []float32, encodedLen, emitStart, emitEnd, frameOffset int64, holdFirst int, resolveSeam func(head []decodedToken) []decodedToken, emit func(delta string)) ([]decodedToken, error) {
	// Acquire a pre-initialized worker. Honor cancellation so a client that
	// disconnects while all workers are busy does not leak a goroutine.
	var w *decoderWorker
	select {
	case w = <-t.decoderPool:
	case <-ctx.Done():
		return nil, ctx.Err()
	}
	// Return the worker to the pool when done. Guard against a panic from
	// sending on a closed pool during shutdown so we never crash the process.
	defer func() {
		defer func() { _ = recover() }()
		t.decoderPool <- w
	}()

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
				dt := decodedToken{id: token, timestep: frameOffset + timestep}
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
