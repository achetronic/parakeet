// Copyright 2025 - Alby Hernández and the parakeet contributors
// SPDX-License-Identifier: Apache-2.0

package server

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"log/slog"
	"net/http"
	"strings"

	"parakeet/internal/asr"
)

// handleStreamingTranscription accepts a request whose body is the raw audio
// bytes (non-multipart), e.g. Content-Type: audio/wav or a chunked upload.
// It buffers the body (capped at 25MB) and returns a single JSON transcript.
// For an SSE delta stream, clients send a multipart request with stream=true
// (handled by streamTranscription in handlers.go).
func (s *Server) handleStreamingTranscription(w http.ResponseWriter, r *http.Request) {
	setCORSHeaders(w)

	if r.Method == "OPTIONS" {
		w.WriteHeader(http.StatusOK)
		return
	}

	if r.Method != http.MethodPost {
		sendError(w, "Method not allowed", "invalid_request_error", http.StatusMethodNotAllowed)
		return
	}

	// 1. Prevent infinite buffer DOS
	r.Body = http.MaxBytesReader(w, r.Body, 25<<20)

	// Determine format
	format := r.URL.Query().Get("format")
	if format == "" {
		// 5. Prevent ffmpeg DOS by avoiding "raw". Default to .wav or try to deduce
		contentType := r.Header.Get("Content-Type")
		switch {
		case strings.Contains(contentType, "audio/mpeg"):
			format = ".mp3"
		case strings.Contains(contentType, "audio/ogg"):
			format = ".ogg"
		case strings.Contains(contentType, "audio/flac"):
			format = ".flac"
		case strings.Contains(contentType, "audio/mp4"):
			format = ".mp4"
		case strings.Contains(contentType, "video/"):
			format = ".mp4"
		default:
			format = ".wav"
		}
	} else if !strings.HasPrefix(format, ".") {
		format = "." + format
	}

	language := r.URL.Query().Get("language")
	if language == "" {
		language = "en" // default
	}

	// Accumulate chunks
	audioData, err := io.ReadAll(r.Body)
	if err != nil {
		sendError(w, "Error reading stream: "+err.Error(), "invalid_request_error", http.StatusBadRequest)
		return
	}

	slog.Info("transcribing streaming audio",
		"bytes", len(audioData),
		"language", language,
		"format", format,
	)

	// 2 & 4. Goroutine leak and deadlock avoided by passing context down to Transcribe
	text, err := s.transcriber.Transcribe(r.Context(), audioData, format, language)
	if err != nil {
		if errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded) {
			return // Context cancelled, ignore
		}
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

	// 3. JSON Injection fixed by using proper encoding
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(TranscriptionResponse{Text: text})
}
