package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"log"
	"net/http"
	"path/filepath"
	"strings"

	"parakeet/internal/asr"
)

// OpenAI-compatible response formats
type TranscriptionResponse struct {
	Text string `json:"text"`
}

type VerboseTranscriptionResponse struct {
	Task     string    `json:"task"`
	Language string    `json:"language"`
	Duration float64   `json:"duration"`
	Text     string    `json:"text"`
	Segments []Segment `json:"segments,omitempty"`
}

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

type ErrorResponse struct {
	Error struct {
		Message string `json:"message"`
		Type    string `json:"type"`
		Code    string `json:"code,omitempty"`
	} `json:"error"`
}

type ModelInfo struct {
	ID      string `json:"id"`
	Object  string `json:"object"`
	Created int64  `json:"created"`
	OwnedBy string `json:"owned_by"`
}

type ModelsResponse struct {
	Object string      `json:"object"`
	Data   []ModelInfo `json:"data"`
}

var transcriber *asr.Transcriber
var debugMode bool

func main() {
	port := flag.Int("port", 5092, "Server port")
	modelsDir := flag.String("models", "./models", "Models directory")
	flag.BoolVar(&debugMode, "debug", false, "Enable debug logging")
	flag.Parse()

	// Enable debug mode in ASR package
	asr.DebugMode = debugMode

	// Initialize transcriber
	var err error
	transcriber, err = asr.NewTranscriber(*modelsDir)
	if err != nil {
		log.Fatalf("Failed to initialize transcriber: %v", err)
	}
	defer transcriber.Close()

	// Setup routes - OpenAI compatible
	http.HandleFunc("/v1/audio/transcriptions", handleTranscription)
	http.HandleFunc("/v1/audio/translations", handleTranslation) // Stub for compatibility
	http.HandleFunc("/v1/models", handleModels)
	http.HandleFunc("/health", handleHealth)

	addr := fmt.Sprintf(":%d", *port)
	log.Printf("🚀 Parakeet ASR server listening on %s", addr)
	log.Printf("📡 POST /v1/audio/transcriptions - OpenAI Whisper-compatible endpoint")
	log.Printf("📋 GET  /v1/models - List available models")
	log.Fatal(http.ListenAndServe(addr, nil))
}

func handleHealth(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]string{"status": "ok"})
}

func handleModels(w http.ResponseWriter, r *http.Request) {
	setCORSHeaders(w)
	if r.Method == "OPTIONS" {
		w.WriteHeader(http.StatusOK)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	resp := ModelsResponse{
		Object: "list",
		Data: []ModelInfo{
			{
				ID:      "parakeet-tdt-0.6b",
				Object:  "model",
				Created: 1700000000,
				OwnedBy: "nvidia",
			},
			{
				ID:      "whisper-1", // Alias for compatibility
				Object:  "model",
				Created: 1700000000,
				OwnedBy: "nvidia",
			},
		},
	}
	json.NewEncoder(w).Encode(resp)
}

func handleTranslation(w http.ResponseWriter, r *http.Request) {
	// Translation endpoint - for now just transcribe (Parakeet is English-focused)
	handleTranscription(w, r)
}

func handleTranscription(w http.ResponseWriter, r *http.Request) {
	setCORSHeaders(w)

	if r.Method == "OPTIONS" {
		w.WriteHeader(http.StatusOK)
		return
	}

	if r.Method != "POST" {
		sendError(w, "Method not allowed", "invalid_request_error", http.StatusMethodNotAllowed)
		return
	}

	// Parse multipart form (25MB max like OpenAI)
	if err := r.ParseMultipartForm(25 << 20); err != nil {
		sendError(w, "Failed to parse form: "+err.Error(), "invalid_request_error", http.StatusBadRequest)
		return
	}

	// Get audio file (required)
	file, header, err := r.FormFile("file")
	if err != nil {
		sendError(w, "Missing required parameter: 'file'", "invalid_request_error", http.StatusBadRequest)
		return
	}
	defer file.Close()

	// Read audio data
	audioData, err := io.ReadAll(file)
	if err != nil {
		sendError(w, "Failed to read audio file: "+err.Error(), "invalid_request_error", http.StatusBadRequest)
		return
	}

	// OpenAI parameters
	model := r.FormValue("model")                    // ignored - we only have one model
	language := r.FormValue("language")              // ISO-639-1 code
	prompt := r.FormValue("prompt")                  // ignored for now
	responseFormat := r.FormValue("response_format") // json, text, srt, verbose_json, vtt
	temperature := r.FormValue("temperature")        // ignored

	_ = model       // Accept but ignore
	_ = prompt      // Accept but ignore
	_ = temperature // Accept but ignore

	// Default response format
	if responseFormat == "" {
		responseFormat = "json"
	}

	// Default language
	if language == "" {
		language = "en"
	}

	log.Printf("Transcribing %s (%d bytes, language=%s, format=%s)",
		header.Filename, len(audioData), language, responseFormat)

	// Determine audio format from extension
	ext := strings.ToLower(filepath.Ext(header.Filename))

	// Transcribe
	text, err := transcriber.Transcribe(audioData, ext, language)
	if err != nil {
		sendError(w, "Transcription failed: "+err.Error(), "server_error", http.StatusInternalServerError)
		return
	}

	if debugMode {
		log.Printf("[DEBUG] Transcription result: %s", text)
	}

	// Calculate approximate duration (16kHz, 16-bit mono)
	duration := float64(len(audioData)) / (16000.0 * 2)

	// Send response based on format
	switch responseFormat {
	case "text":
		w.Header().Set("Content-Type", "text/plain")
		w.Write([]byte(text))

	case "srt":
		w.Header().Set("Content-Type", "text/plain")
		// Simple SRT format
		srt := fmt.Sprintf("1\n00:00:00,000 --> %s\n%s\n", formatSRTTime(duration), text)
		w.Write([]byte(srt))

	case "vtt":
		w.Header().Set("Content-Type", "text/vtt")
		// Simple WebVTT format
		vtt := fmt.Sprintf("WEBVTT\n\n00:00:00.000 --> %s\n%s\n", formatVTTTime(duration), text)
		w.Write([]byte(vtt))

	case "verbose_json":
		w.Header().Set("Content-Type", "application/json")
		resp := VerboseTranscriptionResponse{
			Task:     "transcribe",
			Language: language,
			Duration: duration,
			Text:     text,
			Segments: []Segment{
				{
					ID:               0,
					Seek:             0,
					Start:            0,
					End:              duration,
					Text:             text,
					Tokens:           []int{},
					Temperature:      0,
					AvgLogprob:       -0.5,
					CompressionRatio: 1.0,
					NoSpeechProb:     0.0,
				},
			},
		}
		json.NewEncoder(w).Encode(resp)

	default: // "json"
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(TranscriptionResponse{Text: text})
	}
}

func setCORSHeaders(w http.ResponseWriter) {
	w.Header().Set("Access-Control-Allow-Origin", "*")
	w.Header().Set("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
	w.Header().Set("Access-Control-Allow-Headers", "Content-Type, Authorization, X-Requested-With")
}

func sendError(w http.ResponseWriter, message, errType string, status int) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	resp := ErrorResponse{}
	resp.Error.Message = message
	resp.Error.Type = errType
	json.NewEncoder(w).Encode(resp)
}

func formatSRTTime(seconds float64) string {
	hours := int(seconds) / 3600
	minutes := (int(seconds) % 3600) / 60
	secs := int(seconds) % 60
	millis := int((seconds - float64(int(seconds))) * 1000)
	return fmt.Sprintf("%02d:%02d:%02d,%03d", hours, minutes, secs, millis)
}

func formatVTTTime(seconds float64) string {
	hours := int(seconds) / 3600
	minutes := (int(seconds) % 3600) / 60
	secs := int(seconds) % 60
	millis := int((seconds - float64(int(seconds))) * 1000)
	return fmt.Sprintf("%02d:%02d:%02d.%03d", hours, minutes, secs, millis)
}
