package server

import (
	"fmt"
	"log"
	"net/http"

	"parakeet/internal/asr"
)

// Config holds the server configuration
type Config struct {
	Port      int
	ModelsDir string
	Debug     bool
}

// Server represents the HTTP server for the ASR service
type Server struct {
	config      Config
	transcriber *asr.Transcriber
	mux         *http.ServeMux
}

// New creates a new Server instance with the given configuration
func New(cfg Config) (*Server, error) {
	// Enable debug mode in ASR package
	asr.DebugMode = cfg.Debug

	// Initialize transcriber
	transcriber, err := asr.NewTranscriber(cfg.ModelsDir)
	if err != nil {
		return nil, fmt.Errorf("failed to initialize transcriber: %w", err)
	}

	s := &Server{
		config:      cfg,
		transcriber: transcriber,
		mux:         http.NewServeMux(),
	}

	s.setupRoutes()
	return s, nil
}

// setupRoutes configures the HTTP routes
func (s *Server) setupRoutes() {
	s.mux.HandleFunc("/v1/audio/transcriptions", s.handleTranscription)
	s.mux.HandleFunc("/v1/audio/translations", s.handleTranslation)
	s.mux.HandleFunc("/v1/models", s.handleModels)
	s.mux.HandleFunc("/health", s.handleHealth)
}

// Run starts the HTTP server
func (s *Server) Run() error {
	addr := fmt.Sprintf(":%d", s.config.Port)
	log.Printf("🚀 Parakeet ASR server listening on %s", addr)
	log.Printf("📡 POST /v1/audio/transcriptions - OpenAI Whisper-compatible endpoint")
	log.Printf("📋 GET  /v1/models - List available models")
	return http.ListenAndServe(addr, s.mux)
}

// Close releases server resources
func (s *Server) Close() error {
	if s.transcriber != nil {
		s.transcriber.Close()
	}
	return nil
}
