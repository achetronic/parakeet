package asr

import (
	"bufio"
	"encoding/json"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"

	ort "github.com/yalue/onnxruntime_go"
)

// DebugMode enables verbose logging
var DebugMode bool

type Config struct {
	ModelType         string `json:"model_type"`
	FeaturesSize      int    `json:"features_size"`
	SubsamplingFactor int    `json:"subsampling_factor"`
}

type Transcriber struct {
	config           Config
	vocab            map[int]string
	vocabSize        int
	blankIdx         int
	modelsDir        string
	maxTokensPerStep int
	mel              *MelFilterbank
}

func NewTranscriber(modelsDir string) (*Transcriber, error) {
	t := &Transcriber{
		modelsDir:        modelsDir,
		maxTokensPerStep: 10,
		blankIdx:         8192, // <blk> token
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

	// Set defaults
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
		// Try common locations
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

	if libPath != "" {
		ort.SetSharedLibraryPath(libPath)
		if err := ort.InitializeEnvironment(); err != nil {
			return nil, fmt.Errorf("failed to initialize ONNX Runtime: %w", err)
		}
	} else {
		return nil, fmt.Errorf("ONNX Runtime library not found. Set ONNXRUNTIME_LIB env var or install libonnxruntime")
	}

	// Verify model files exist
	encoderPath := filepath.Join(modelsDir, "encoder-model.int8.onnx")
	if _, err := os.Stat(encoderPath); os.IsNotExist(err) {
		encoderPath = filepath.Join(modelsDir, "encoder-model.onnx")
		if _, err := os.Stat(encoderPath); os.IsNotExist(err) {
			return nil, fmt.Errorf("encoder model not found. Download from https://huggingface.co/istupakov/parakeet-tdt-0.6b-v3-onnx")
		}
	}

	decoderPath := filepath.Join(modelsDir, "decoder_joint-model.int8.onnx")
	if _, err := os.Stat(decoderPath); os.IsNotExist(err) {
		decoderPath = filepath.Join(modelsDir, "decoder_joint-model.onnx")
		if _, err := os.Stat(decoderPath); os.IsNotExist(err) {
			return nil, fmt.Errorf("decoder model not found. Download from https://huggingface.co/istupakov/parakeet-tdt-0.6b-v3-onnx")
		}
	}

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
		// Replace SentencePiece marker with space
		token = strings.ReplaceAll(token, "▁", " ")
		t.vocab[id] = token
		if token == "<blk>" {
			t.blankIdx = id
		}
	}
	t.vocabSize = len(t.vocab)

	if DebugMode {
		log.Printf("[DEBUG] Vocab loaded: %d tokens, blankIdx=%d", t.vocabSize, t.blankIdx)
	}

	return scanner.Err()
}

func (t *Transcriber) Close() {
	ort.DestroyEnvironment()
}

func (t *Transcriber) Transcribe(audioData []byte, format, language string) (string, error) {
	// Convert audio to float32 waveform at 16kHz
	waveform, err := t.loadAudio(audioData, format)
	if err != nil {
		return "", fmt.Errorf("failed to load audio: %w", err)
	}

	if DebugMode {
		log.Printf("[DEBUG] Waveform length: %d samples (%.2f seconds)", len(waveform), float64(len(waveform))/16000.0)
	}

	if len(waveform) < 1600 { // Less than 100ms
		if DebugMode {
			log.Printf("[DEBUG] Audio too short: %d samples", len(waveform))
		}
		return "", nil
	}

	// Extract mel features
	features := t.mel.Extract(waveform)
	if len(features) == 0 {
		return "", fmt.Errorf("no features extracted")
	}

	if DebugMode {
		log.Printf("[DEBUG] Mel features: %d frames x %d features", len(features), len(features[0]))
	}

	// Run inference
	tokens, err := t.runInference(features)
	if err != nil {
		return "", fmt.Errorf("inference failed: %w", err)
	}

	if DebugMode {
		log.Printf("[DEBUG] Tokens decoded: %d tokens = %v", len(tokens), tokens)
	}

	// Convert tokens to text
	text := t.tokensToText(tokens)
	return text, nil
}

func (t *Transcriber) loadAudio(data []byte, format string) ([]float32, error) {
	switch format {
	case ".wav":
		return parseWAV(data)
	case ".webm", ".ogg", ".mp3", ".m4a":
		return nil, fmt.Errorf("format %s requires ffmpeg conversion - not yet implemented", format)
	default:
		// Try to parse as WAV
		return parseWAV(data)
	}
}

func (t *Transcriber) runInference(features [][]float32) ([]int, error) {
	// Prepare input tensor - shape: [batch, features, time]
	batchSize := int64(1)
	numFeatures := int64(t.config.FeaturesSize)
	numFrames := int64(len(features))

	// Flatten features to [1, features, frames] format (transposed from [frames, features])
	inputData := make([]float32, numFeatures*numFrames)
	for f := int64(0); f < numFrames; f++ {
		for m := int64(0); m < numFeatures && m < int64(len(features[f])); m++ {
			inputData[m*numFrames+f] = features[f][m]
		}
	}

	// Create input tensors
	inputShape := ort.NewShape(batchSize, numFeatures, numFrames)
	inputTensor, err := ort.NewTensor(inputShape, inputData)
	if err != nil {
		return nil, fmt.Errorf("failed to create input tensor: %w", err)
	}
	defer inputTensor.Destroy()

	lengthData := []int64{numFrames}
	lengthShape := ort.NewShape(batchSize)
	lengthTensor, err := ort.NewTensor(lengthShape, lengthData)
	if err != nil {
		return nil, fmt.Errorf("failed to create length tensor: %w", err)
	}
	defer lengthTensor.Destroy()

	// Encoder output shape: [batch, time/subsampling, encoder_dim]
	// Estimate output size
	encodedLen := (numFrames-1)/int64(t.config.SubsamplingFactor) + 1
	encoderDim := int64(1024) // Typical for Conformer models

	outputShape := ort.NewShape(batchSize, encoderDim, encodedLen)
	outputTensor, err := ort.NewEmptyTensor[float32](outputShape)
	if err != nil {
		return nil, fmt.Errorf("failed to create output tensor: %w", err)
	}
	defer outputTensor.Destroy()

	outLenShape := ort.NewShape(batchSize)
	outLenTensor, err := ort.NewEmptyTensor[int64](outLenShape)
	if err != nil {
		return nil, fmt.Errorf("failed to create output length tensor: %w", err)
	}
	defer outLenTensor.Destroy()

	// Load and run encoder
	encoderPath := filepath.Join(t.modelsDir, "encoder-model.int8.onnx")
	if _, err := os.Stat(encoderPath); os.IsNotExist(err) {
		encoderPath = filepath.Join(t.modelsDir, "encoder-model.onnx")
	}

	encoderSession, err := ort.NewAdvancedSession(
		encoderPath,
		[]string{"audio_signal", "length"},
		[]string{"outputs", "encoded_lengths"},
		[]ort.ArbitraryTensor{inputTensor, lengthTensor},
		[]ort.ArbitraryTensor{outputTensor, outLenTensor},
		nil,
	)
	if err != nil {
		return nil, fmt.Errorf("failed to create encoder session: %w", err)
	}
	defer encoderSession.Destroy()

	if err := encoderSession.Run(); err != nil {
		return nil, fmt.Errorf("encoder run failed: %w", err)
	}

	// Get encoder outputs
	encoderOut := outputTensor.GetData()
	actualEncodedLen := outLenTensor.GetData()[0]

	if DebugMode {
		log.Printf("[DEBUG] Encoder output: %d floats, actualEncodedLen=%d", len(encoderOut), actualEncodedLen)
	}

	// Now run TDT decoder
	tokens, err := t.tdtDecode(encoderOut, encoderDim, actualEncodedLen)
	if err != nil {
		return nil, fmt.Errorf("decoding failed: %w", err)
	}

	return tokens, nil
}

func (t *Transcriber) tdtDecode(encoderOut []float32, encoderDim, encodedLen int64) ([]int, error) {
	decoderPath := filepath.Join(t.modelsDir, "decoder_joint-model.int8.onnx")
	if _, err := os.Stat(decoderPath); os.IsNotExist(err) {
		decoderPath = filepath.Join(t.modelsDir, "decoder_joint-model.onnx")
	}

	if DebugMode {
		log.Printf("[DEBUG] TDT decode: encoderOut len=%d, encoderDim=%d, encodedLen=%d", len(encoderOut), encoderDim, encodedLen)
	}

	// Decoder state dimensions (from model inspection)
	stateDim := int64(640)
	numLayers := int64(2)

	var tokens []int
	timestep := int64(0)
	emittedTokens := 0
	prevToken := t.blankIdx

	// Initialize states
	state1 := make([]float32, numLayers*1*stateDim)
	state2 := make([]float32, numLayers*1*stateDim)

	for timestep < encodedLen {
		// Extract encoder output at current timestep
		// Shape: [1, encoder_dim, 1]
		encOutSlice := make([]float32, encoderDim)
		for d := int64(0); d < encoderDim; d++ {
			idx := d*encodedLen + timestep
			if idx < int64(len(encoderOut)) {
				encOutSlice[d] = encoderOut[idx]
			}
		}

		// Create decoder input tensors
		encOutTensor, err := ort.NewTensor(ort.NewShape(1, encoderDim, 1), encOutSlice)
		if err != nil {
			return nil, err
		}

		targetsTensor, err := ort.NewTensor(ort.NewShape(1, 1), []int32{int32(prevToken)})
		if err != nil {
			encOutTensor.Destroy()
			return nil, err
		}

		targetLenTensor, err := ort.NewTensor(ort.NewShape(1), []int32{1})
		if err != nil {
			encOutTensor.Destroy()
			targetsTensor.Destroy()
			return nil, err
		}

		state1Tensor, err := ort.NewTensor(ort.NewShape(numLayers, 1, stateDim), state1)
		if err != nil {
			encOutTensor.Destroy()
			targetsTensor.Destroy()
			targetLenTensor.Destroy()
			return nil, err
		}

		state2Tensor, err := ort.NewTensor(ort.NewShape(numLayers, 1, stateDim), state2)
		if err != nil {
			encOutTensor.Destroy()
			targetsTensor.Destroy()
			targetLenTensor.Destroy()
			state1Tensor.Destroy()
			return nil, err
		}

		// Output tensors
		// TDT output includes vocab logits + duration logits
		// Shape: [batch, target_len, 1, vocab_size + num_duration_classes]
		// For Parakeet TDT: vocab_size=8193, num_duration_classes=5, total=8198
		numDurationClasses := int64(5) // TDT uses 5 duration classes (0-4)
		outputDim := int64(t.vocabSize) + numDurationClasses
		outputTensor, err := ort.NewEmptyTensor[float32](ort.NewShape(1, 1, 1, outputDim))
		if err != nil {
			encOutTensor.Destroy()
			targetsTensor.Destroy()
			targetLenTensor.Destroy()
			state1Tensor.Destroy()
			state2Tensor.Destroy()
			return nil, err
		}

		outState1Tensor, err := ort.NewEmptyTensor[float32](ort.NewShape(numLayers, 1, stateDim))
		if err != nil {
			encOutTensor.Destroy()
			targetsTensor.Destroy()
			targetLenTensor.Destroy()
			state1Tensor.Destroy()
			state2Tensor.Destroy()
			outputTensor.Destroy()
			return nil, err
		}

		outState2Tensor, err := ort.NewEmptyTensor[float32](ort.NewShape(numLayers, 1, stateDim))
		if err != nil {
			encOutTensor.Destroy()
			targetsTensor.Destroy()
			targetLenTensor.Destroy()
			state1Tensor.Destroy()
			state2Tensor.Destroy()
			outputTensor.Destroy()
			outState1Tensor.Destroy()
			return nil, err
		}

		// Create decoder session
		decoderSession, err := ort.NewAdvancedSession(
			decoderPath,
			[]string{"encoder_outputs", "targets", "target_length", "input_states_1", "input_states_2"},
			[]string{"outputs", "output_states_1", "output_states_2"},
			[]ort.ArbitraryTensor{encOutTensor, targetsTensor, targetLenTensor, state1Tensor, state2Tensor},
			[]ort.ArbitraryTensor{outputTensor, outState1Tensor, outState2Tensor},
			nil,
		)
		if err != nil {
			encOutTensor.Destroy()
			targetsTensor.Destroy()
			targetLenTensor.Destroy()
			state1Tensor.Destroy()
			state2Tensor.Destroy()
			outputTensor.Destroy()
			outState1Tensor.Destroy()
			outState2Tensor.Destroy()
			return nil, fmt.Errorf("failed to create decoder session: %w", err)
		}

		if err := decoderSession.Run(); err != nil {
			decoderSession.Destroy()
			encOutTensor.Destroy()
			targetsTensor.Destroy()
			targetLenTensor.Destroy()
			state1Tensor.Destroy()
			state2Tensor.Destroy()
			outputTensor.Destroy()
			outState1Tensor.Destroy()
			outState2Tensor.Destroy()
			return nil, fmt.Errorf("decoder run failed: %w", err)
		}

		// Get outputs
		output := outputTensor.GetData()

		// TDT: first vocabSize elements are token logits, rest are duration logits
		vocabLogits := output[:t.vocabSize]
		durationLogits := output[t.vocabSize:]

		// Find best token (greedy)
		token := argmax(vocabLogits)

		// Find best duration step
		step := argmax(durationLogits)

		if DebugMode && timestep < 5 {
			log.Printf("[DEBUG] t=%d: token=%d (blank=%d), step=%d, maxLogit=%.3f", timestep, token, t.blankIdx, step, vocabLogits[token])
		}

		if token != t.blankIdx {
			// Update states
			copy(state1, outState1Tensor.GetData())
			copy(state2, outState2Tensor.GetData())
			tokens = append(tokens, token)
			prevToken = token
			emittedTokens++
		}

		// Advance timestep based on TDT duration prediction
		if step > 0 {
			timestep += int64(step)
			emittedTokens = 0
		} else if token == t.blankIdx || emittedTokens >= t.maxTokensPerStep {
			timestep++
			emittedTokens = 0
		}

		// Cleanup
		decoderSession.Destroy()
		encOutTensor.Destroy()
		targetsTensor.Destroy()
		targetLenTensor.Destroy()
		state1Tensor.Destroy()
		state2Tensor.Destroy()
		outputTensor.Destroy()
		outState1Tensor.Destroy()
		outState2Tensor.Destroy()
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
			// Skip special tokens
			if strings.HasPrefix(text, "<") && strings.HasSuffix(text, ">") {
				continue
			}
			parts = append(parts, text)
		}
	}
	text := strings.Join(parts, "")

	// Clean up spacing
	re := regexp.MustCompile(`^\s+|\s+$|\s{2,}`)
	text = re.ReplaceAllStringFunc(text, func(s string) string {
		if strings.TrimSpace(s) == "" {
			return " "
		}
		return s
	})
	text = strings.TrimSpace(text)
	return text
}
