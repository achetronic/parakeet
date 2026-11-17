// SPDX-FileCopyrightText: 2026 Alby Hernández <hola@achetronic.com>
// SPDX-License-Identifier: Apache-2.0

package server

// TranscriptionResponse represents a simple transcription result
type TranscriptionResponse struct {
	Text string `json:"text"`
}

// VerboseTranscriptionResponse represents a detailed transcription result.
// Segments has no omitempty: verbose_json clients expect the key to exist even
// when the audio produced no speech and the list is empty.
type VerboseTranscriptionResponse struct {
	Task     string    `json:"task"`
	Language string    `json:"language"`
	Duration float64   `json:"duration"`
	Text     string    `json:"text"`
	Segments []Segment `json:"segments"`
}

// SpeakerTurn is a speaker-attributed transcription segment.
type SpeakerTurn struct {
	Speaker int     `json:"speaker"`
	Start   float64 `json:"start"`
	End     float64 `json:"end"`
	Text    string  `json:"text"`
}

// DiarizedTranscriptionResponse is returned when diarize=true.
type DiarizedTranscriptionResponse struct {
	Task     string        `json:"task"`
	Language string        `json:"language"`
	Duration float64       `json:"duration"`
	Text     string        `json:"text"`
	Segments []SpeakerTurn `json:"segments"`
}

// Segment represents a transcription segment with timing information.
// Start and End are seconds; Seek is the segment start in milliseconds, matching
// OpenAI's verbose_json field. AvgLogprob is the mean log probability the decoder
// assigned to this segment's tokens. CompressionRatio is the gzip ratio of the
// segment text, the same metric OpenAI reports. NoSpeechProb stays 0 because the
// TDT decoder has no no-speech head.
type Segment struct {
	ID               int     `json:"id"`
	Seek             int     `json:"seek"`
	Start            float64 `json:"start"`
	End              float64 `json:"end"`
	Text             string  `json:"text"`
	Tokens           []int   `json:"tokens"`
	Temperature      float64 `json:"temperature"`
	AvgLogprob       float64 `json:"avg_logprob"`
	CompressionRatio float64 `json:"compression_ratio"`
	NoSpeechProb     float64 `json:"no_speech_prob"`
}

// StreamDeltaEvent is emitted (as SSE) for each chunk of transcript produced
// while the model is still decoding. Mirrors OpenAI's transcript.text.delta.
type StreamDeltaEvent struct {
	Type  string `json:"type"` // always "transcript.text.delta"
	Delta string `json:"delta"`
}

// StreamDoneEvent is the final SSE event, carrying the complete transcript.
// Mirrors OpenAI's transcript.text.done.
type StreamDoneEvent struct {
	Type string `json:"type"` // always "transcript.text.done"
	Text string `json:"text"`
}

// ErrorResponse represents an OpenAI-compatible error response
type ErrorResponse struct {
	Error ErrorDetail `json:"error"`
}

// ErrorDetail contains error information
type ErrorDetail struct {
	Message string `json:"message"`
	Type    string `json:"type"`
	Code    string `json:"code,omitempty"`
}

// ModelInfo represents information about an available model.
// Loaded reports whether the weight sessions are in memory: an idle or manual
// unload reports false, and the next request reloads them on demand.
type ModelInfo struct {
	ID      string `json:"id"`
	Object  string `json:"object"`
	Created int64  `json:"created"`
	OwnedBy string `json:"owned_by"`
	Loaded  bool   `json:"loaded"`
}

// ModelStateResponse is the answer of the load and unload endpoints: one entry
// per model the server holds, with its state once the call is done. Models whose
// feature is disabled are not listed.
type ModelStateResponse struct {
	Object string       `json:"object"` // always "model.state"
	Models []ModelState `json:"models"`
}

// ModelState is one model. Loaded is the state after the call, so an unload
// always reports false and a load always reports true.
type ModelState struct {
	ID     string `json:"id"`
	Loaded bool   `json:"loaded"`
}

// ModelsResponse represents the list of available models
type ModelsResponse struct {
	Object string      `json:"object"`
	Data   []ModelInfo `json:"data"`
}
