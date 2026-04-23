package main

import (
	"context"
	"flag"
	"log/slog"
	"os"
	"os/signal"
	"strings"
	"syscall"
	"time"

	"parakeet/internal/server"
)

func main() {
	cfg := server.Config{}

	flag.IntVar(&cfg.Port, "port", 5092, "Server port")
	flag.StringVar(&cfg.ModelsDir, "models", "./models", "Models directory")
	flag.StringVar(&cfg.LogLevel, "log-level", "info", "Log level: debug, info, warn, error")
	flag.StringVar(&cfg.LogFormat, "log-format", "text", "Log format: text or json")
	flag.IntVar(&cfg.Workers, "workers", 4, "Number of concurrent inference workers (each uses ~670MB RAM for int8 models)")
	flag.BoolVar(&cfg.FFmpegEnabled, "ffmpeg", true, "Enable ffmpeg fallback for non-WAV audio (requires ffmpeg in PATH)")
	flag.StringVar(&cfg.FFmpegPath, "ffmpeg-path", "", "Path to the ffmpeg binary (default: resolved from PATH)")
	flag.DurationVar(&cfg.FFmpegTimeout, "ffmpeg-timeout", 60*time.Second, "Maximum wall-clock time for a single ffmpeg conversion")
	flag.Parse()

	setupLogger(cfg.LogFormat, cfg.LogLevel)

	srv, err := server.New(cfg)
	if err != nil {
		slog.Error("failed to create server", "error", err)
		os.Exit(1)
	}

	// Run server in background
	errCh := make(chan error, 1)
	go func() {
		errCh <- srv.Run()
	}()

	// Wait for shutdown signal or server error
	quit := make(chan os.Signal, 1)
	signal.Notify(quit, syscall.SIGINT, syscall.SIGTERM)

	select {
	case sig := <-quit:
		slog.Info("received signal, shutting down", "signal", sig)
	case err := <-errCh:
		if err != nil {
			slog.Error("server error", "error", err)
			srv.Close()
			os.Exit(1)
		}
	}

	// Graceful shutdown: wait up to 30s for in-flight requests to finish
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	if err := srv.Shutdown(ctx); err != nil {
		slog.Error("shutdown error", "error", err)
	}

	srv.Close()
	slog.Info("server stopped")
}

func setupLogger(format, level string) {
	var slogLevel slog.Level
	switch strings.ToLower(level) {
	case "debug":
		slogLevel = slog.LevelDebug
	case "warn":
		slogLevel = slog.LevelWarn
	case "error":
		slogLevel = slog.LevelError
	default:
		slogLevel = slog.LevelInfo
	}

	opts := &slog.HandlerOptions{Level: slogLevel}

	var handler slog.Handler
	switch format {
	case "json":
		handler = slog.NewJSONHandler(os.Stdout, opts)
	default:
		handler = slog.NewTextHandler(os.Stdout, opts)
	}

	slog.SetDefault(slog.New(handler))
}
