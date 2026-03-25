package server

import (
	"fmt"
	"log/slog"
	"net/http"
	"os"
	"strings"

	"parakeet/internal/asr"
)

const apiKeyEnvVar = "PARAKEET_API_KEY"

// Config holds the server configuration
type Config struct {
	Port      int
	ModelsDir string
	LogLevel  string
	LogFormat string
}

// Server represents the HTTP server for the ASR service
type Server struct {
	config      Config
	transcriber *asr.Transcriber
	mux         *http.ServeMux
	apiKey      string
}

// New creates a new Server instance with the given configuration
func New(cfg Config) (*Server, error) {
	// Enable debug mode in ASR package
	asr.DebugMode = cfg.LogLevel == "debug"

	// Initialize transcriber
	transcriber, err := asr.NewTranscriber(cfg.ModelsDir)
	if err != nil {
		return nil, fmt.Errorf("failed to initialize transcriber: %w", err)
	}

	s := &Server{
		config:      cfg,
		transcriber: transcriber,
		mux:         http.NewServeMux(),
		apiKey:      os.Getenv(apiKeyEnvVar),
	}

	if s.apiKey != "" {
		slog.Info("API key authentication enabled")
	}

	s.setupRoutes()
	return s, nil
}

// setupRoutes configures the HTTP routes
func (s *Server) setupRoutes() {
	s.mux.HandleFunc("/v1/audio/transcriptions", s.requireAuth(s.handleTranscription))
	s.mux.HandleFunc("/v1/audio/translations", s.requireAuth(s.handleTranslation))
	s.mux.HandleFunc("/v1/models", s.requireAuth(s.handleModels))
	s.mux.HandleFunc("/health", s.handleHealth)
}

// requireAuth wraps a handler with API key authentication.
// If no API key is configured, requests pass through without checks.
func (s *Server) requireAuth(next http.HandlerFunc) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if s.apiKey == "" {
			next(w, r)
			return
		}

		auth := r.Header.Get("Authorization")
		token := strings.TrimPrefix(auth, "Bearer ")
		if auth == "" || token != s.apiKey {
			sendError(w, "Invalid API key", "authentication_error", http.StatusUnauthorized)
			return
		}

		next(w, r)
	}
}

// Run starts the HTTP server
func (s *Server) Run() error {
	addr := fmt.Sprintf(":%d", s.config.Port)
	slog.Info("Parakeet ASR server started", "addr", addr)
	slog.Info("endpoints registered",
		"transcriptions", "POST /v1/audio/transcriptions",
		"models", "GET /v1/models",
	)
	return http.ListenAndServe(addr, s.mux)
}

// Close releases server resources
func (s *Server) Close() error {
	if s.transcriber != nil {
		s.transcriber.Close()
	}
	return nil
}
