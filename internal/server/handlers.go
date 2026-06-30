// Copyright 2025 - Alby Hernández and the parakeet contributors
// SPDX-License-Identifier: Apache-2.0

package server

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"path/filepath"
	"strings"
	"time"

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

// handleTranscription routes to either multipart or streaming handler based on Content-Type
func (s *Server) handleTranscription(w http.ResponseWriter, r *http.Request) {
	if strings.HasPrefix(r.Header.Get("Content-Type"), "multipart/form-data") {
		s.handleMultipartTranscription(w, r)
	} else {
		s.handleStreamingTranscription(w, r)
	}
}

// handleMultipartTranscription handles audio transcription requests via multipart form
func (s *Server) handleMultipartTranscription(w http.ResponseWriter, r *http.Request) {
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
	streamRequested := parseBool(r.FormValue("stream"))

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

	slog.Info("transcribing",
		"file", header.Filename,
		"bytes", len(audioData),
		"language", language,
		"format", responseFormat,
	)

	// Determine audio format from extension
	ext := strings.ToLower(filepath.Ext(header.Filename))

	// Streaming path: emit SSE transcript.text.delta events as the decoder
	// produces text, then a final transcript.text.done. Only json/text
	// formats are streamable; others fall through to the buffered path.
	if streamRequested && (responseFormat == "json" || responseFormat == "text") {
		s.streamTranscription(w, r, audioData, ext, language)
		return
	}

	// Transcribe
	text, err := s.transcriber.Transcribe(r.Context(), audioData, ext, language)
	if err != nil {
		// Unsupported or malformed audio is a client error: the request
		// body we received cannot be decoded. Everything else is treated
		// as an internal failure.
		if errors.Is(err, asr.ErrUnsupportedAudio) {
			sendError(w, "Unsupported or malformed audio: "+err.Error(), "invalid_request_error", http.StatusBadRequest)
			return
		}
		sendError(w, "Transcription failed: "+err.Error(), "server_error", http.StatusInternalServerError)
		return
	}

	if asr.DebugMode {
		slog.Debug("transcription result", "text", text)
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

// parseBool interprets common truthy form values ("true", "1", "yes", "on").
func parseBool(v string) bool {
	switch strings.ToLower(strings.TrimSpace(v)) {
	case "true", "1", "yes", "on":
		return true
	}
	return false
}

// streamTranscription transcribes audioData and streams the result to the
// client as Server-Sent Events, following OpenAI's streaming transcription
// protocol: a series of transcript.text.delta events followed by a single
// transcript.text.done event carrying the full transcript.
func (s *Server) streamTranscription(w http.ResponseWriter, r *http.Request, audioData []byte, ext, language string) {
	flusher, ok := w.(http.Flusher)
	if !ok {
		// The ResponseWriter cannot stream; degrade gracefully to a buffered
		// JSON response so the client still gets a valid result.
		text, err := s.transcriber.Transcribe(r.Context(), audioData, ext, language)
		if err != nil {
			s.writeTranscribeError(w, err)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(TranscriptionResponse{Text: text})
		return
	}

	// SSE headers must be set before the first write / WriteHeader.
	w.Header().Set("Content-Type", "text/event-stream; charset=utf-8")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	// Disable proxy buffering (nginx) so events reach the client immediately.
	w.Header().Set("X-Accel-Buffering", "no")
	w.WriteHeader(http.StatusOK)
	flusher.Flush()

	// ResponseController lets us set a per-write deadline. This is what makes
	// a slow/stalled reader recoverable: if the client stops draining its TCP
	// receive window, the write below fails instead of blocking forever inside
	// the decoder goroutine (which holds a worker). On failure we cancel the
	// context so the decoder stops and releases its worker. We deliberately do
	// NOT use a global http.Server WriteTimeout, which would kill healthy long
	// streams; the deadline is reset before every event instead.
	rc := http.NewResponseController(w)
	const writeDeadline = 30 * time.Second

	// Derive a cancelable context: if a write to the client fails (disconnect,
	// broken pipe, stalled reader past the deadline), we cancel so the decoder
	// stops promptly and releases its worker instead of computing into the void.
	ctx, cancel := context.WithCancel(r.Context())
	defer cancel()

	// writeEvent serializes one SSE frame: "event: <type>\ndata: <json>\n\n".
	// Each event is marshaled independently so a mid-write failure can never
	// corrupt a subsequent frame. Returns false on write failure.
	writeEvent := func(eventType string, v interface{}) bool {
		payload, err := json.Marshal(v)
		if err != nil {
			return false
		}
		// Bound how long a single event may take to flush to the client. A
		// reader that has stalled will trip this deadline and the write fails,
		// freeing the worker (slow-reader DoS mitigation).
		_ = rc.SetWriteDeadline(time.Now().Add(writeDeadline))
		if _, err := fmt.Fprintf(w, "event: %s\ndata: %s\n\n", eventType, payload); err != nil {
			cancel()
			return false
		}
		if err := rc.Flush(); err != nil {
			cancel()
			return false
		}
		return true
	}

	text, err := s.transcriber.TranscribeStream(ctx, audioData, ext, language, func(delta string) {
		writeEvent("transcript.text.delta", StreamDeltaEvent{Type: "transcript.text.delta", Delta: delta})
	})
	if err != nil {
		// Headers (200 OK) are already sent, so we cannot switch to an HTTP
		// error status. Client cancellation needs no payload (nobody is
		// listening); any other failure is surfaced as a terminal SSE error.
		if errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded) {
			return
		}
		msg := "Transcription failed: " + err.Error()
		errType := "server_error"
		if errors.Is(err, asr.ErrUnsupportedAudio) {
			msg = "Unsupported or malformed audio: " + err.Error()
			errType = "invalid_request_error"
		}
		writeEvent("error", ErrorResponse{Error: ErrorDetail{Message: msg, Type: errType}})
		return
	}

	writeEvent("transcript.text.done", StreamDoneEvent{Type: "transcript.text.done", Text: text})
}

// writeTranscribeError maps a transcription error to an OpenAI-compatible HTTP
// error response. Only safe to call before any body has been written.
func (s *Server) writeTranscribeError(w http.ResponseWriter, err error) {
	if errors.Is(err, asr.ErrUnsupportedAudio) {
		sendError(w, "Unsupported or malformed audio: "+err.Error(), "invalid_request_error", http.StatusBadRequest)
		return
	}
	sendError(w, "Transcription failed: "+err.Error(), "server_error", http.StatusInternalServerError)
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
