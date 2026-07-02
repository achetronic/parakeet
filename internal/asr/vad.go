// SPDX-FileCopyrightText: 2026 Alby Hernández <hola@achetronic.com>
// SPDX-License-Identifier: Apache-2.0

package asr

import (
	"fmt"
	"log/slog"
	"os"

	ort "github.com/yalue/onnxruntime_go"
)

// Silero VAD constants. These describe the pinned v6.2.1 ONNX contract
// (snakers4/silero-vad, MIT). The model is stateful: its recurrent state travels
// through the "state"/"stateN" tensors per call rather than living inside the
// session, so a single session can be shared across concurrent requests as long
// as every caller keeps its own vadState.
const (
	// vadWindowSamples is the fixed analysis window Silero v5+ expects at 16 kHz:
	// 512 samples, i.e. 32 ms per inference.
	vadWindowSamples = 512

	// vadContextSamples is the left-context Silero v5+ prepends to each window.
	// The 16 kHz model consumes vadContextSamples+vadWindowSamples (576) samples
	// per call; the trailing vadContextSamples of one window become the context
	// of the next. The first window uses zero context.
	vadContextSamples = 64

	// vadStateElements is the flattened size of the recurrent state tensor,
	// shaped [2, 1, 128].
	vadStateElements = 2 * 1 * 128

	// vadSampleRate is the only sample rate this integration feeds the model.
	// The rest of the pipeline is already 16 kHz mono (see audio.go).
	vadSampleRate int64 = 16000
)

// vadState carries one request's Silero recurrent state and left-context between
// sequential window inferences. Each transcription owns its own vadState, so the
// shared session stays safe under concurrency. Reset zeroes it, which the
// boundary oracle does at the start of every overlap region.
type vadState struct {
	state [vadStateElements]float32
	ctx   [vadContextSamples]float32
}

// reset clears the recurrent state and left-context so the next window starts a
// fresh analysis. The boundary oracle resets before each overlap region.
func (s *vadState) reset() {
	for i := range s.state {
		s.state[i] = 0
	}
	for i := range s.ctx {
		s.ctx[i] = 0
	}
}

// sileroVAD wraps the shared Silero VAD ONNX session. Like the encoder,
// it is a single long-lived DynamicAdvancedSession reused across requests and
// runs OUTSIDE the decoder worker pool, so its concurrency is bounded by the
// number of in-flight HTTP requests rather than by -workers.
type sileroVAD struct {
	session *ort.DynamicAdvancedSession
	srData  []int64
}

// newSileroVAD loads the Silero VAD model from path and creates the shared
// session. A missing file is NOT fatal: the caller logs a warning once and the
// boundary stack degrades to mel energy. Any other error (corrupt model, ORT
// failure) is returned so it surfaces loudly at startup.
func newSileroVAD(path string, sessOpts *ort.SessionOptions) (*sileroVAD, error) {
	if _, err := os.Stat(path); os.IsNotExist(err) {
		return nil, os.ErrNotExist
	}

	session, err := ort.NewDynamicAdvancedSession(
		path,
		[]string{"input", "state", "sr"},
		[]string{"output", "stateN"},
		sessOpts,
	)
	if err != nil {
		return nil, fmt.Errorf("create Silero VAD session: %w", err)
	}

	return &sileroVAD{
		session: session,
		srData:  []int64{vadSampleRate},
	}, nil
}

// destroy releases the underlying ONNX session.
func (v *sileroVAD) destroy() {
	if v != nil && v.session != nil {
		v.session.Destroy()
		v.session = nil
	}
}

// infer runs one window through the model and returns the speech probability in
// [0, 1]. It prepends st.ctx as left context, updates st.ctx with the trailing
// samples of window, and advances the recurrent state in st.state. window must
// be exactly vadWindowSamples long.
func (v *sileroVAD) infer(st *vadState, window []float32) (float32, error) {
	if len(window) != vadWindowSamples {
		return 0, fmt.Errorf("silero window must be %d samples, got %d", vadWindowSamples, len(window))
	}

	// Assemble context + window into the model input.
	input := make([]float32, vadContextSamples+vadWindowSamples)
	copy(input, st.ctx[:])
	copy(input[vadContextSamples:], window)
	// Save the trailing samples as context for the next window.
	copy(st.ctx[:], window[vadWindowSamples-vadContextSamples:])

	inputTensor, err := ort.NewTensor(ort.NewShape(1, int64(len(input))), input)
	if err != nil {
		return 0, fmt.Errorf("create VAD input tensor: %w", err)
	}
	defer inputTensor.Destroy()

	// The state tensor is written in place across calls, so give the model a
	// copy of the current state and read the fresh state from the output.
	stateData := make([]float32, vadStateElements)
	copy(stateData, st.state[:])
	stateTensor, err := ort.NewTensor(ort.NewShape(2, 1, 128), stateData)
	if err != nil {
		return 0, fmt.Errorf("create VAD state tensor: %w", err)
	}
	defer stateTensor.Destroy()

	srTensor, err := ort.NewTensor(ort.NewShape(1), v.srData)
	if err != nil {
		return 0, fmt.Errorf("create VAD sr tensor: %w", err)
	}
	defer srTensor.Destroy()

	outputTensor, err := ort.NewEmptyTensor[float32](ort.NewShape(1, 1))
	if err != nil {
		return 0, fmt.Errorf("create VAD output tensor: %w", err)
	}
	defer outputTensor.Destroy()

	stateOutTensor, err := ort.NewEmptyTensor[float32](ort.NewShape(2, 1, 128))
	if err != nil {
		return 0, fmt.Errorf("create VAD state output tensor: %w", err)
	}
	defer stateOutTensor.Destroy()

	if err := v.session.Run(
		[]ort.Value{inputTensor, stateTensor, srTensor},
		[]ort.Value{outputTensor, stateOutTensor},
	); err != nil {
		return 0, fmt.Errorf("VAD run failed: %w", err)
	}

	copy(st.state[:], stateOutTensor.GetData())

	prob := outputTensor.GetData()[0]
	return prob, nil
}

// speechProbabilities runs the model over samples in sequential windows and
// returns one speech probability per full window. It resets st first so each
// overlap region is analysed independently. Trailing samples that do not fill a
// full window are ignored. On any inference error it logs and returns what it
// has, so a VAD hiccup degrades to fewer probabilities rather than failing the
// whole transcription.
func (v *sileroVAD) speechProbabilities(st *vadState, samples []float32) []float32 {
	st.reset()

	numWindows := len(samples) / vadWindowSamples
	if numWindows == 0 {
		return nil
	}

	probs := make([]float32, 0, numWindows)
	for i := 0; i < numWindows; i++ {
		window := samples[i*vadWindowSamples : (i+1)*vadWindowSamples]
		prob, err := v.infer(st, window)
		if err != nil {
			slog.Warn("VAD inference failed mid-overlap, using partial probabilities",
				"window", i, "error", err)
			break
		}
		probs = append(probs, prob)
	}
	return probs
}
