// SPDX-FileCopyrightText: 2026 Alby Hernández <hola@achetronic.com>
// SPDX-License-Identifier: Apache-2.0

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

// envPrefix namespaces every environment variable derived from a command-line flag.
const envPrefix = "PARAKEET_"

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
	flag.StringVar(&cfg.GPUProvider, "gpu", "cpu", "Execution provider: cpu or cuda")
	flag.IntVar(&cfg.GPUDeviceID, "gpu-device", 0, "GPU device index for cuda")
	flag.IntVar(&cfg.ChunkSeconds, "chunk-seconds", 300, "Sliding-window size in seconds for long audio (must stay under the model limit)")
	flag.IntVar(&cfg.ChunkOverlapSeconds, "chunk-overlap-seconds", 15, "Overlap in seconds between consecutive chunks")
	flag.Parse()

	// Any flag not set on the command line falls back to its matching env var,
	// e.g. --log-level -> PARAKEET_LOG_LEVEL. Precedence: CLI flag > env var > default.
	applyEnvDefaults(flag.CommandLine)

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

// applyEnvDefaults sources any flag not passed explicitly on the command line from
// its matching environment variable, mapping the flag name to upper snake case with
// the PARAKEET_ prefix (e.g. --log-level -> PARAKEET_LOG_LEVEL). This gives every
// flag an env var for free, so new flags need no extra wiring. Precedence stays
// CLI flag > env var > flag default: flags set on the CLI are skipped, and the
// value is parsed through the flag's own type so an invalid value is rejected
// (with a warning) instead of silently corrupting the config.
func applyEnvDefaults(fs *flag.FlagSet) {
	// Flags set explicitly on the CLI win and must not be overridden by env.
	setOnCLI := make(map[string]bool)
	fs.Visit(func(f *flag.Flag) { setOnCLI[f.Name] = true })

	fs.VisitAll(func(f *flag.Flag) {
		if setOnCLI[f.Name] {
			return
		}
		key := envPrefix + strings.ToUpper(strings.ReplaceAll(f.Name, "-", "_"))
		val, ok := os.LookupEnv(key)
		if !ok {
			return
		}
		// Snapshot the current value: flag.Value.Set clobbers numeric flags to zero
		// even when parsing fails, so restore it on error to keep the default.
		prev := f.Value.String()
		if err := f.Value.Set(val); err != nil {
			slog.Warn("ignoring invalid environment variable",
				"var", key, "value", val, "error", err)
			_ = f.Value.Set(prev)
		}
	})
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
