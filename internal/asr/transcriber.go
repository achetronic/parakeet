package asr

import (
	"bufio"
	"encoding/json"
	"fmt"
	"log/slog"
	"os"
	"os/exec"
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

func newDecoderWorker(decoderPath string, vocabSize int) (*decoderWorker, error) {
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
		nil,
	)
	if err != nil {
		w.destroy()
		return nil, fmt.Errorf("create decoder session: %w", err)
	}

	return w, nil
}

type Transcriber struct {
	config           Config
	vocab            map[int]string
	vocabSize        int
	blankIdx         int
	maxTokensPerStep int
	mel              *MelFilterbank
	encoderPath      string
	decoderPool      chan *decoderWorker
}

func NewTranscriber(modelsDir string, workers int) (*Transcriber, error) {
	t := &Transcriber{
		maxTokensPerStep: 10,
		blankIdx:         8192,
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

	// Initialize ONNX Runtime
	libPath := os.Getenv("ONNXRUNTIME_LIB")
	if libPath == "" {
		commonPaths := []string{
			"/usr/lib/libonnxruntime.so",
			"/usr/lib/x86_64-linux-gnu/libonnxruntime.so",
			"/usr/local/lib/libonnxruntime.so",
			"/opt/onnxruntime/lib/libonnxruntime.so",
			"./libonnxruntime.so",
			"libonnxruntime.so.1.17.0",
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

	// Resolve encoder path (stored, reused per request)
	t.encoderPath = filepath.Join(modelsDir, "encoder-model.int8.onnx")
	if _, err := os.Stat(t.encoderPath); os.IsNotExist(err) {
		t.encoderPath = filepath.Join(modelsDir, "encoder-model.onnx")
		if _, err := os.Stat(t.encoderPath); os.IsNotExist(err) {
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

	// Create decoder worker pool — each worker owns a persistent session and
	// pre-allocated tensors. Workers are acquired per request and returned after.
	if workers < 1 {
		workers = 1
	}
	t.decoderPool = make(chan *decoderWorker, workers)
	for i := 0; i < workers; i++ {
		w, err := newDecoderWorker(decoderPath, t.vocabSize)
		if err != nil {
			t.Close()
			return nil, fmt.Errorf("failed to create decoder worker %d: %w", i, err)
		}
		t.decoderPool <- w
	}

	slog.Info("transcriber initialized",
		"workers", workers,
		"encoder", filepath.Base(t.encoderPath),
		"decoder", filepath.Base(decoderPath),
		"vocabSize", t.vocabSize,
	)

	return t, nil
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

// Close releases all pool workers and the ONNX Runtime environment.
func (t *Transcriber) Close() {
	if t.decoderPool != nil {
		close(t.decoderPool)
		for w := range t.decoderPool {
			w.destroy()
		}
	}
	ort.DestroyEnvironment()
}

func (t *Transcriber) Transcribe(audioData []byte, format, language string) (string, error) {
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

	tokens, err := t.runInference(features)
	if err != nil {
		return "", fmt.Errorf("inference failed: %w", err)
	}

	if DebugMode {
		slog.Debug("tokens decoded", "count", len(tokens), "tokens", tokens)
	}

	return t.tokensToText(tokens), nil
}

func (t *Transcriber) loadAudio(data []byte, format string) ([]float32, error) {
	switch format {
	case ".wav":
		return parseWAV(data)
	case ".webm", ".ogg", ".mp3", ".m4a", ".flac", ".aac", ".wma", ".opus":
		return convertWithFFmpeg(data, format)
	default:
		samples, err := parseWAV(data)
		if err == nil {
			return samples, nil
		}
		return convertWithFFmpeg(data, format)
	}
}

func convertWithFFmpeg(data []byte, format string) ([]float32, error) {
	tempDir := os.TempDir()
	inputPath := filepath.Join(tempDir, fmt.Sprintf("parakeet_%d%s", os.Getpid(), format))
	outputPath := filepath.Join(tempDir, fmt.Sprintf("parakeet_%d.wav", os.Getpid()))

	if err := os.WriteFile(inputPath, data, 0600); err != nil {
		return nil, fmt.Errorf("write temp input: %w", err)
	}
	defer os.Remove(inputPath)
	defer os.Remove(outputPath)

	cmd := exec.Command("ffmpeg", "-y", "-i", inputPath,
		"-ac", "1", "-ar", "16000", "-acodec", "pcm_s16le", outputPath)

	if err := cmd.Run(); err != nil {
		return nil, fmt.Errorf("ffmpeg: %w", err)
	}

	wavData, err := os.ReadFile(outputPath)
	if err != nil {
		return nil, fmt.Errorf("read converted: %w", err)
	}

	return parseWAV(wavData)
}

func (t *Transcriber) runInference(features [][]float32) ([]int, error) {
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

	// Encoder session is created per request because input shape varies with audio length.
	// The session object loads the model graph; the file is OS-cached after first request.
	encoderSession, err := ort.NewAdvancedSession(
		t.encoderPath,
		[]string{"audio_signal", "length"},
		[]string{"outputs", "encoded_lengths"},
		[]ort.ArbitraryTensor{inputTensor, lengthTensor},
		[]ort.ArbitraryTensor{outputTensor, outLenTensor},
		nil,
	)
	if err != nil {
		return nil, fmt.Errorf("create encoder session: %w", err)
	}
	defer encoderSession.Destroy()

	if err := encoderSession.Run(); err != nil {
		return nil, fmt.Errorf("encoder run failed: %w", err)
	}

	encoderOut := outputTensor.GetData()
	actualEncodedLen := outLenTensor.GetData()[0]

	if DebugMode {
		slog.Debug("encoder output", "floats", len(encoderOut), "encodedLen", actualEncodedLen)
	}

	// Decoder tensors (encoderOut) must remain alive during tdtDecode.
	// The defers above fire after tdtDecode returns, so this is safe.
	return t.tdtDecode(encoderOut, actualEncodedLen)
}

func (t *Transcriber) tdtDecode(encoderOut []float32, encodedLen int64) ([]int, error) {
	// Acquire a pre-initialized worker. Blocks if all workers are busy.
	w := <-t.decoderPool
	defer func() { t.decoderPool <- w }()

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

	var tokens []int
	timestep := int64(0)
	emittedTokens := 0
	prevToken := t.blankIdx

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
			tokens = append(tokens, token)
			prevToken = token
			emittedTokens++
		}

		if step > 0 {
			timestep += int64(step)
			emittedTokens = 0
		} else if token == t.blankIdx || emittedTokens >= t.maxTokensPerStep {
			timestep++
			emittedTokens = 0
		}
	}

	return tokens, nil
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

func (t *Transcriber) tokensToText(tokens []int) string {
	var parts []string
	for _, tok := range tokens {
		if text, ok := t.vocab[tok]; ok {
			if strings.HasPrefix(text, "<") && strings.HasSuffix(text, ">") {
				continue
			}
			parts = append(parts, text)
		}
	}
	text := strings.Join(parts, "")
	text = strings.TrimSpace(whitespaceRegex.ReplaceAllString(text, " "))
	return text
}
