package main

import (
	"flag"
	"log/slog"
	"os"
	"strings"

	"parakeet/internal/server"
)

func main() {
	cfg := server.Config{}

	flag.IntVar(&cfg.Port, "port", 5092, "Server port")
	flag.StringVar(&cfg.ModelsDir, "models", "./models", "Models directory")
	flag.StringVar(&cfg.LogLevel, "log-level", "info", "Log level: debug, info, warn, error")
	flag.StringVar(&cfg.LogFormat, "log-format", "text", "Log format: text or json")
	flag.Parse()

	setupLogger(cfg.LogFormat, cfg.LogLevel)

	srv, err := server.New(cfg)
	if err != nil {
		slog.Error("failed to create server", "error", err)
		os.Exit(1)
	}
	defer srv.Close()

	if err := srv.Run(); err != nil {
		slog.Error("server error", "error", err)
		os.Exit(1)
	}
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
