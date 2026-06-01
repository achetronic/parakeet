package server

// TranscriptionResponse represents a simple transcription result
type TranscriptionResponse struct {
	Text string `json:"text"`
}

// VerboseTranscriptionResponse represents a detailed transcription result
type VerboseTranscriptionResponse struct {
	Task     string    `json:"task"`
	Language string    `json:"language"`
	Duration float64   `json:"duration"`
	Text     string    `json:"text"`
	Segments []Segment `json:"segments,omitempty"`
}

// Segment represents a transcription segment with timing information
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

// ModelInfo represents information about an available model
type ModelInfo struct {
	ID      string `json:"id"`
	Object  string `json:"object"`
	Created int64  `json:"created"`
	OwnedBy string `json:"owned_by"`
}

// ModelsResponse represents the list of available models
type ModelsResponse struct {
	Object string      `json:"object"`
	Data   []ModelInfo `json:"data"`
}
