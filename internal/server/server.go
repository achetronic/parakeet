package server

import (
	"context"
	"fmt"
	"log/slog"
	"net/http"
	"os"
	"strings"
	"time"

	"parakeet/internal/asr"
)

const apiKeyEnvVar = "PARAKEET_API_KEY"

// Config holds the server configuration
type Config struct {
	Port      int
	ModelsDir string
	LogLevel  string
	LogFormat string
	Workers   int

	// FFmpegEnabled toggles the ffmpeg-backed fallback for non-WAV audio.
	// When true, unknown input formats are transcoded to 16 kHz mono WAV
	// before transcription. When false, only WAV input is accepted.
	FFmpegEnabled bool

	// FFmpegPath is the name or absolute path of the ffmpeg binary.
	// Empty means "ffmpeg", resolved against PATH.
	FFmpegPath string

	// FFmpegTimeout bounds the duration of a single conversion.
	FFmpegTimeout time.Duration
}

// Server represents the HTTP server for the ASR service
type Server struct {
	config      Config
	transcriber *asr.Transcriber
	httpServer  *http.Server
	mux         *http.ServeMux
	apiKey      string
}

// New creates a new Server instance with the given configuration
func New(cfg Config) (*Server, error) {
	// Enable debug mode in ASR package
	asr.DebugMode = cfg.LogLevel == "debug"

	// Initialize transcriber
	transcriber, err := asr.NewTranscriber(cfg.ModelsDir, cfg.Workers, asr.Options{
		FFmpeg: asr.FFmpegConfig{
			Enabled:    cfg.FFmpegEnabled,
			BinaryPath: cfg.FFmpegPath,
			Timeout:    cfg.FFmpegTimeout,
		},
	})
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

// Run starts the HTTP server. It blocks until the server is shut down.
// Returns nil if closed via Shutdown; returns the underlying error otherwise.
func (s *Server) Run() error {
	addr := fmt.Sprintf(":%d", s.config.Port)
	s.httpServer = &http.Server{
		Addr:    addr,
		Handler: s.mux,
	}
	slog.Info("Parakeet ASR server started", "addr", addr)
	slog.Info("endpoints registered",
		"transcriptions", "POST /v1/audio/transcriptions",
		"models", "GET /v1/models",
	)
	err := s.httpServer.ListenAndServe()
	if err == http.ErrServerClosed {
		return nil
	}
	return err
}

// Shutdown gracefully stops the HTTP server, waiting for in-flight requests
// to complete before returning. After Shutdown returns, all request handlers
// have finished and it is safe to call Close.
func (s *Server) Shutdown(ctx context.Context) error {
	if s.httpServer != nil {
		slog.Info("shutting down HTTP server, waiting for in-flight requests...")
		return s.httpServer.Shutdown(ctx)
	}
	return nil
}

// Close releases server resources. Must be called after Shutdown.
func (s *Server) Close() error {
	if s.transcriber != nil {
		s.transcriber.Close()
	}
	return nil
}
