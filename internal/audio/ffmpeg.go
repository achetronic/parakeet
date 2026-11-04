// SPDX-FileCopyrightText: 2026 Alby Hernández <hola@achetronic.com>
// SPDX-License-Identifier: Apache-2.0

package audio

import (
	"bytes"
	"context"
	"fmt"
	"log/slog"
	"os"
	"os/exec"
	"time"
)

// FFmpegConfig controls optional ffmpeg-backed conversion of non-WAV inputs.
type FFmpegConfig struct {
	Enabled    bool
	BinaryPath string
	Timeout    time.Duration
}

// Converter transcodes audio through an external ffmpeg binary. It is safe
// for concurrent use: each Convert call uses its own temporary files.
type Converter struct {
	binaryPath string
	timeout    time.Duration
}

// NewConverter returns a ready-to-use converter, or nil when ffmpeg is
// unavailable. A nil converter is not an error; it means non-WAV inputs are
// rejected with ErrUnsupportedAudio.
func NewConverter(cfg FFmpegConfig) *Converter {
	if !cfg.Enabled {
		return nil
	}

	bin := cfg.BinaryPath
	if bin == "" {
		bin = "ffmpeg"
	}
	resolved, err := exec.LookPath(bin)
	if err != nil {
		slog.Warn("ffmpeg not found, non-WAV inputs will be rejected", "requested", bin, "error", err)
		return nil
	}

	timeout := cfg.Timeout
	if timeout <= 0 {
		timeout = 60 * time.Second
	}

	slog.Info("ffmpeg conversion enabled", "binary", resolved, "timeout", timeout)
	return &Converter{binaryPath: resolved, timeout: timeout}
}

// Convert transcodes arbitrary audio bytes into a 16 kHz mono PCM WAV payload.
func (c *Converter) Convert(data []byte) ([]byte, error) {
	if c == nil {
		return nil, ErrUnsupportedAudio
	}

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
	out.Close()
	defer os.Remove(outputPath)

	ctx, cancel := context.WithTimeout(context.Background(), c.timeout)
	defer cancel()

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
		return nil, fmt.Errorf("ffmpeg: %s: %w", trimStderr(stderr.String()), ErrUnsupportedAudio)
	}

	wavData, err := os.ReadFile(outputPath)
	if err != nil {
		return nil, fmt.Errorf("ffmpeg: read converted output: %w", err)
	}

	if Debug {
		slog.Debug("ffmpeg conversion succeeded", "inputBytes", len(data), "outputBytes", len(wavData))
	}
	return wavData, nil
}

// trimStderr shortens ffmpeg stderr to a single line with a sensible cap.
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
	for len(out) > 0 && out[len(out)-1] == ' ' {
		out = out[:len(out)-1]
	}
	return string(out)
}
