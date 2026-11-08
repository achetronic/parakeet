// SPDX-FileCopyrightText: 2026 Alby Hernández <hola@achetronic.com>
// SPDX-License-Identifier: Apache-2.0

package server

import (
	"context"
	"fmt"
	"log/slog"
	"net/http"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"parakeet/internal/asr"
	"parakeet/internal/audio"
	"parakeet/internal/diarization"
)

const apiKeyEnvVar = "PARAKEET_API_KEY"

// ParseGPUDevices splits a comma-separated list of CUDA device ids ("0,1,2")
// into an int slice. Empty input yields nil. It rejects blank entries and
// anything that is not a non-negative integer.
func ParseGPUDevices(s string) ([]int, error) {
	s = strings.TrimSpace(s)
	if s == "" {
		return nil, nil
	}
	var devices []int
	for _, part := range strings.Split(s, ",") {
		id := strings.TrimSpace(part)
		if id == "" {
			return nil, fmt.Errorf("empty device id in %q", s)
		}
		n, err := strconv.Atoi(id)
		if err != nil || n < 0 {
			return nil, fmt.Errorf("invalid device id %q", id)
		}
		devices = append(devices, n)
	}
	return devices, nil
}

// Config holds the server configuration
type Config struct {
	Port      int
	ModelsDir string
	LogLevel  string
	LogFormat string
	Workers   int

	// ModelPrecision selects the ONNX precision: "int8" (default) or "fp32".
	// It picks models/<family>/<precision> under ModelsDir.
	ModelPrecision string

	// Attention selects the ASR encoder: "full" (default) or "local"
	// (rel_pos_local_attn). Local lifts the single-pass positional-table limit.
	Attention string

	// FFmpegEnabled toggles the ffmpeg-backed fallback for non-WAV audio.
	// When true, unknown input formats are transcoded to 16 kHz mono WAV
	// before transcription. When false, only WAV input is accepted.
	FFmpegEnabled bool

	// FFmpegPath is the name or absolute path of the ffmpeg binary.
	// Empty means "ffmpeg", resolved against PATH.
	FFmpegPath string

	// FFmpegTimeout bounds the duration of a single conversion.
	FFmpegTimeout time.Duration

	// GPUProvider selects the ONNX Runtime execution provider: "cpu" (default)
	// or "cuda". An unknown value fails fast at startup.
	GPUProvider string

	// GPUDevices lists the CUDA devices to spread inference workers over, e.g.
	// {0, 1, 2}. Ignored on CPU. Empty means a single device.
	GPUDevices []int

	// ChunkSeconds is the sliding-window size for long audio, in seconds.
	// ChunkOverlapSeconds is how much consecutive windows share so words at
	// the seams keep their context. LongAudio enables the windowing; when off,
	// audio over the model limit is rejected instead of chunked.
	ChunkSeconds        int
	ChunkOverlapSeconds int
	LongAudio           bool

	// DisableVADBasedChunking and DisableMelBasedChunking turn off the first two
	// layers of the chunk-boundary cascade (Silero VAD, then mel energy). The
	// arithmetic midpoint is always the final fallback. VADModelPath overrides
	// where the Silero VAD model is loaded from; empty means silero_vad.onnx
	// inside the models directory.
	DisableVADBasedChunking bool
	DisableMelBasedChunking bool
	VADModelPath            string

	// EnableDiarization loads the Nemotron-3-Diarization model at startup and
	// enables the diarize=true transcription parameter. Off by default so
	// Parakeet-only deployments work without the extra model.
	EnableDiarization bool
}

// Server represents the HTTP server for the ASR service
type Server struct {
	config      Config
	converter   *audio.Converter
	transcriber *asr.Transcriber
	diarizer    *diarization.Diarizer
	httpServer  *http.Server
	mux         *http.ServeMux
	apiKey      string
}

// New creates a new Server instance with the given configuration
func New(cfg Config) (*Server, error) {
	// Enable debug mode in the ASR and audio packages.
	asr.DebugMode = cfg.LogLevel == "debug"
	audio.Debug = cfg.LogLevel == "debug"

	provider, err := asr.ParseProvider(cfg.GPUProvider)
	if err != nil {
		return nil, err
	}

	if cfg.ModelPrecision != "int8" && cfg.ModelPrecision != "fp32" {
		return nil, fmt.Errorf("invalid model precision %q: must be int8 or fp32", cfg.ModelPrecision)
	}
	if cfg.Attention != "" && cfg.Attention != "full" && cfg.Attention != "local" {
		return nil, fmt.Errorf("invalid attention %q: must be full or local", cfg.Attention)
	}

	converter := audio.NewConverter(audio.FFmpegConfig{
		Enabled:    cfg.FFmpegEnabled,
		BinaryPath: cfg.FFmpegPath,
		Timeout:    cfg.FFmpegTimeout,
	})

	// Initialize transcriber
	asrDir := filepath.Join(cfg.ModelsDir, "asr", cfg.ModelPrecision)
	vadPath := cfg.VADModelPath
	if vadPath == "" {
		vadPath = filepath.Join(cfg.ModelsDir, "vad", "silero_vad.onnx")
	}
	transcriber, err := asr.NewTranscriber(asrDir, cfg.Workers, asr.Options{
		FFmpeg: converter,
		GPU: asr.GPUConfig{
			Provider: provider,
			Devices:  cfg.GPUDevices,
		},
		Chunk: asr.ChunkConfig{
			Enabled:        cfg.LongAudio,
			Seconds:        cfg.ChunkSeconds,
			OverlapSeconds: cfg.ChunkOverlapSeconds,
		},
		Boundary: asr.BoundaryConfig{
			DisableVAD:   cfg.DisableVADBasedChunking,
			DisableMel:   cfg.DisableMelBasedChunking,
			VADModelPath: vadPath,
		},
		Attention: cfg.Attention,
	})
	if err != nil {
		return nil, fmt.Errorf("failed to initialize transcriber: %w", err)
	}

	// Initialize the diarizer only when speaker diarization is enabled (opt-in
	// via -enable-diarization). It is off by default so Parakeet-only
	// deployments do not need the extra model.
	var diarizer *diarization.Diarizer
	if cfg.EnableDiarization {
		d, err := diarization.New(filepath.Join(cfg.ModelsDir, "diarization", cfg.ModelPrecision), cfg.Workers)
		if err != nil {
			transcriber.Close()
			return nil, fmt.Errorf("failed to initialize diarizer: %w", err)
		}
		diarizer = d
	}

	s := &Server{
		config:      cfg,
		converter:   converter,
		transcriber: transcriber,
		diarizer:    diarizer,
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
		// ReadHeaderTimeout bounds the time to read request headers, defending
		// against Slowloris without capping the body upload or the response.
		// We intentionally do NOT set WriteTimeout: streaming (SSE) responses
		// are long-lived and a global write deadline would cut them off.
		ReadHeaderTimeout: 30 * time.Second,
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
	if s.diarizer != nil {
		s.diarizer.Close()
	}
	return nil
}
