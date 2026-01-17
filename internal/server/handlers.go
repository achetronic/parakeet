package server

import (
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"path/filepath"
	"strings"

	"parakeet/internal/asr"
)

// handleHealth returns the server health status
func (s *Server) handleHealth(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]string{"status": "ok"})
}

// handleModels returns the list of available models
func (s *Server) handleModels(w http.ResponseWriter, r *http.Request) {
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

// handleTranslation handles translation requests (delegates to transcription for now)
func (s *Server) handleTranslation(w http.ResponseWriter, r *http.Request) {
	// Translation endpoint - for now just transcribe (Parakeet is English-focused)
	s.handleTranscription(w, r)
}

// handleTranscription handles audio transcription requests
func (s *Server) handleTranscription(w http.ResponseWriter, r *http.Request) {
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
	text, err := s.transcriber.Transcribe(audioData, ext, language)
	if err != nil {
		sendError(w, "Transcription failed: "+err.Error(), "server_error", http.StatusInternalServerError)
		return
	}

	if asr.DebugMode {
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

// setCORSHeaders sets CORS headers for cross-origin requests
func setCORSHeaders(w http.ResponseWriter) {
	w.Header().Set("Access-Control-Allow-Origin", "*")
	w.Header().Set("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
	w.Header().Set("Access-Control-Allow-Headers", "Content-Type, Authorization, X-Requested-With")
}

// sendError sends an OpenAI-compatible error response
func sendError(w http.ResponseWriter, message, errType string, status int) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	resp := ErrorResponse{
		Error: ErrorDetail{
			Message: message,
			Type:    errType,
		},
	}
	json.NewEncoder(w).Encode(resp)
}

// formatSRTTime formats duration as SRT timestamp
func formatSRTTime(seconds float64) string {
	hours := int(seconds) / 3600
	minutes := (int(seconds) % 3600) / 60
	secs := int(seconds) % 60
	millis := int((seconds - float64(int(seconds))) * 1000)
	return fmt.Sprintf("%02d:%02d:%02d,%03d", hours, minutes, secs, millis)
}

// formatVTTTime formats duration as WebVTT timestamp
func formatVTTTime(seconds float64) string {
	hours := int(seconds) / 3600
	minutes := (int(seconds) % 3600) / 60
	secs := int(seconds) % 60
	millis := int((seconds - float64(int(seconds))) * 1000)
	return fmt.Sprintf("%02d:%02d:%02d.%03d", hours, minutes, secs, millis)
}
