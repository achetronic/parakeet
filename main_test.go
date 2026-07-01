// SPDX-FileCopyrightText: 2026 Alby Hernández <hola@achetronic.com>
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"flag"
	"testing"
	"time"
)

// newTestFlags builds an isolated FlagSet mirroring the real flags so tests never
// touch the global flag.CommandLine.
func newTestFlags() (*flag.FlagSet, *struct {
	port    int
	level   string
	ffmpeg  bool
	timeout time.Duration
}) {
	fs := flag.NewFlagSet("test", flag.ContinueOnError)
	vals := &struct {
		port    int
		level   string
		ffmpeg  bool
		timeout time.Duration
	}{}
	fs.IntVar(&vals.port, "port", 5092, "")
	fs.StringVar(&vals.level, "log-level", "info", "")
	fs.BoolVar(&vals.ffmpeg, "ffmpeg", true, "")
	fs.DurationVar(&vals.timeout, "ffmpeg-timeout", 60*time.Second, "")
	return fs, vals
}

func TestApplyEnvDefaults(t *testing.T) {
	t.Run("env value fills a flag left at its default", func(t *testing.T) {
		t.Setenv("PARAKEET_PORT", "8080")
		fs, vals := newTestFlags()
		if err := fs.Parse(nil); err != nil {
			t.Fatalf("parse: %v", err)
		}
		applyEnvDefaults(fs)
		if vals.port != 8080 {
			t.Fatalf("port = %d, want 8080 (from env)", vals.port)
		}
	})

	t.Run("explicit CLI flag beats the env var", func(t *testing.T) {
		t.Setenv("PARAKEET_PORT", "8080")
		fs, vals := newTestFlags()
		if err := fs.Parse([]string{"-port", "9090"}); err != nil {
			t.Fatalf("parse: %v", err)
		}
		applyEnvDefaults(fs)
		if vals.port != 9090 {
			t.Fatalf("port = %d, want 9090 (CLI overrides env)", vals.port)
		}
	})

	t.Run("no env keeps the flag default", func(t *testing.T) {
		fs, vals := newTestFlags()
		if err := fs.Parse(nil); err != nil {
			t.Fatalf("parse: %v", err)
		}
		applyEnvDefaults(fs)
		if vals.port != 5092 {
			t.Fatalf("port = %d, want 5092 (default)", vals.port)
		}
	})

	t.Run("dashed flag name maps to upper snake case env var", func(t *testing.T) {
		t.Setenv("PARAKEET_LOG_LEVEL", "debug")
		t.Setenv("PARAKEET_FFMPEG_TIMEOUT", "30s")
		fs, vals := newTestFlags()
		if err := fs.Parse(nil); err != nil {
			t.Fatalf("parse: %v", err)
		}
		applyEnvDefaults(fs)
		if vals.level != "debug" {
			t.Fatalf("log-level = %q, want %q", vals.level, "debug")
		}
		if vals.timeout != 30*time.Second {
			t.Fatalf("ffmpeg-timeout = %s, want 30s", vals.timeout)
		}
	})

	t.Run("typed flag parses env value through its own type", func(t *testing.T) {
		t.Setenv("PARAKEET_FFMPEG", "false")
		fs, vals := newTestFlags()
		if err := fs.Parse(nil); err != nil {
			t.Fatalf("parse: %v", err)
		}
		applyEnvDefaults(fs)
		if vals.ffmpeg {
			t.Fatal("ffmpeg = true, want false (from env)")
		}
	})

	t.Run("invalid env value is ignored and the default survives", func(t *testing.T) {
		t.Setenv("PARAKEET_PORT", "not-a-number")
		fs, vals := newTestFlags()
		if err := fs.Parse(nil); err != nil {
			t.Fatalf("parse: %v", err)
		}
		applyEnvDefaults(fs)
		if vals.port != 5092 {
			t.Fatalf("port = %d, want 5092 (invalid env ignored)", vals.port)
		}
	})
}
