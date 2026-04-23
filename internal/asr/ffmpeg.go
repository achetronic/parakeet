package asr

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"log/slog"
	"os"
	"os/exec"
	"time"
)

// ErrUnsupportedAudio is returned when the input is neither a parsable WAV
// nor convertible via ffmpeg (either because ffmpeg is disabled/missing or
// because ffmpeg itself rejected the input). Callers can use errors.Is to
// detect this condition and map it to HTTP 400.
var ErrUnsupportedAudio = errors.New("unsupported audio")

// FFmpegConfig controls optional ffmpeg-backed conversion of non-WAV inputs.
//
// When Enabled is true, loadAudio will attempt to transcode unknown inputs
// to 16 kHz mono PCM WAV via an external ffmpeg binary. When Enabled is
// false (the default outside of environments where ffmpeg was found), only
// WAV input is accepted.
type FFmpegConfig struct {
	// Enabled toggles ffmpeg-backed conversion.
	Enabled bool

	// BinaryPath is the resolved absolute path to the ffmpeg executable.
	BinaryPath string

	// Timeout bounds the wall-clock time of a single conversion.
	Timeout time.Duration
}

// ffmpegConverter performs audio transcoding using an external ffmpeg binary.
//
// It is concurrency-safe: each call to Convert writes to its own temporary
// input and output files created via os.CreateTemp, so simultaneous requests
// never share paths. This matters because the decoder worker pool allows up
// to `-workers` inferences in parallel, and each of them may be preceded by
// a conversion.
type ffmpegConverter struct {
	binaryPath string
	timeout    time.Duration
}

// newFFmpegConverter returns a ready-to-use converter or nil when ffmpeg is
// unavailable. A nil converter is not an error; it means non-WAV inputs will
// be rejected with ErrUnsupportedAudio. The probing is done once at startup
// to fail fast and surface a clear log line instead of discovering the
// problem on the first request.
func newFFmpegConverter(cfg FFmpegConfig) *ffmpegConverter {
	if !cfg.Enabled {
		return nil
	}

	bin := cfg.BinaryPath
	if bin == "" {
		bin = "ffmpeg"
	}

	resolved, err := exec.LookPath(bin)
	if err != nil {
		slog.Warn("ffmpeg not found, non-WAV inputs will be rejected",
			"requested", bin,
			"error", err,
		)
		return nil
	}

	timeout := cfg.Timeout
	if timeout <= 0 {
		timeout = 60 * time.Second
	}

	slog.Info("ffmpeg conversion enabled",
		"binary", resolved,
		"timeout", timeout,
	)

	return &ffmpegConverter{
		binaryPath: resolved,
		timeout:    timeout,
	}
}

// Convert transcodes arbitrary audio bytes into 16 kHz mono PCM WAV bytes
// by shelling out to ffmpeg. It returns the raw WAV payload so the caller
// can feed it into parseWAV and reuse the existing decode path.
//
// The function is safe for concurrent use: it allocates unique temporary
// files for each invocation and cleans them up on return.
func (c *ffmpegConverter) Convert(data []byte) ([]byte, error) {
	if c == nil {
		return nil, ErrUnsupportedAudio
	}

	// Unique temp files per call. os.CreateTemp randomizes the suffix so
	// concurrent workers never collide on disk.
	in, err := os.CreateTemp("", "parakeet-in-*.bin")
	if err != nil {
		return nil, fmt.Errorf("ffmpeg: create temp input: %w", err)
	}
	inputPath := in.Name()
	defer os.Remove(inputPath)

	if _, err := in.Write(data); err != nil {
		in.Close()
		return nil, fmt.Errorf("ffmpeg: write temp input: %w", err)
	}
	if err := in.Close(); err != nil {
		return nil, fmt.Errorf("ffmpeg: close temp input: %w", err)
	}

	out, err := os.CreateTemp("", "parakeet-out-*.wav")
	if err != nil {
		return nil, fmt.Errorf("ffmpeg: create temp output: %w", err)
	}
	outputPath := out.Name()
	// Close the file handle immediately; ffmpeg will rewrite it.
	out.Close()
	defer os.Remove(outputPath)

	ctx, cancel := context.WithTimeout(context.Background(), c.timeout)
	defer cancel()

	// -nostdin: never read from stdin (defensive, avoids hangs).
	// -y: overwrite output without prompting.
	// -hide_banner -loglevel error: keep stderr focused on real errors.
	// -ac 1 -ar 16000 -acodec pcm_s16le: match the pipeline expectation.
	// -f wav: force WAV container regardless of output filename.
	cmd := exec.CommandContext(ctx, c.binaryPath,
		"-nostdin",
		"-hide_banner",
		"-loglevel", "error",
		"-y",
		"-i", inputPath,
		"-ac", "1",
		"-ar", "16000",
		"-acodec", "pcm_s16le",
		"-f", "wav",
		outputPath,
	)

	var stderr bytes.Buffer
	cmd.Stderr = &stderr

	if err := cmd.Run(); err != nil {
		if ctx.Err() == context.DeadlineExceeded {
			return nil, fmt.Errorf("ffmpeg: conversion timed out after %s: %w", c.timeout, ErrUnsupportedAudio)
		}
		// ffmpeg exited non-zero: the input is either corrupted or in a
		// format ffmpeg can't decode. Treat both as client-side errors.
		return nil, fmt.Errorf("ffmpeg: %s: %w", trimStderr(stderr.String()), ErrUnsupportedAudio)
	}

	wavData, err := os.ReadFile(outputPath)
	if err != nil {
		return nil, fmt.Errorf("ffmpeg: read converted output: %w", err)
	}

	if DebugMode {
		slog.Debug("ffmpeg conversion succeeded",
			"inputBytes", len(data),
			"outputBytes", len(wavData),
		)
	}

	return wavData, nil
}

// trimStderr shortens ffmpeg stderr to a single line with a sensible cap so
// it fits in an HTTP error response without leaking a wall of text.
func trimStderr(s string) string {
	s = stripNewlines(s)
	const maxLen = 200
	if len(s) > maxLen {
		return s[:maxLen] + "..."
	}
	if s == "" {
		return "conversion failed"
	}
	return s
}

func stripNewlines(s string) string {
	out := make([]byte, 0, len(s))
	for i := 0; i < len(s); i++ {
		c := s[i]
		if c == '\n' || c == '\r' {
			if len(out) > 0 && out[len(out)-1] != ' ' {
				out = append(out, ' ')
			}
			continue
		}
		out = append(out, c)
	}
	// Trim trailing space.
	for len(out) > 0 && out[len(out)-1] == ' ' {
		out = out[:len(out)-1]
	}
	return string(out)
}
