// SPDX-FileCopyrightText: 2026 Alby Hernández <hola@achetronic.com>
// SPDX-License-Identifier: Apache-2.0

//go:build seaminspect

// Package asr's seam inspector. Build-tag gated so it never runs in the normal
// test suite (it needs ONNX Runtime, the models, and the reference audio). No
// Python anywhere in this repo: this is the Go equivalent of a scratch script.
//
// It transcribes the reference MP3 through the full long-audio pipeline, logs
// the chosen chunk-boundary positions (enable debug to see which oracle layer
// decided each one), and prints the transcribed words around every seam next to
// the reference .srt text for the same time range, so a human can eyeball each
// seam for the duplicated/dropped words from issue #18. It needs no network.
//
// Usage:
//
//	PARAKEET_MODELS=./models \
//	PARAKEET_SEAM_AUDIO=./testdata/reference/learn-case-interviews.mp3 \
//	PARAKEET_SEAM_SRT=./testdata/reference/learn-case-interviews.srt \
//	go test -tags=seaminspect -run TestSeamInspection -v ./internal/asr/
package asr

import (
	"bufio"
	"context"
	"log/slog"
	"os"
	"sort"
	"strconv"
	"strings"
	"testing"
	"time"
)

// seamInspectWindowSeconds is how far either side of a seam we quote text.
const seamInspectWindowSeconds = 5.0

func TestSeamInspection(t *testing.T) {
	audioPath := os.Getenv("PARAKEET_SEAM_AUDIO")
	if audioPath == "" {
		audioPath = "../../testdata/reference/learn-case-interviews.mp3"
	}
	srtPath := os.Getenv("PARAKEET_SEAM_SRT")
	if srtPath == "" {
		srtPath = "../../testdata/reference/learn-case-interviews.srt"
	}
	modelsDir := os.Getenv("PARAKEET_MODELS")
	if modelsDir == "" {
		modelsDir = "../../models"
	}

	if _, err := os.Stat(audioPath); err != nil {
		t.Skipf("reference audio not found (%v); nothing to inspect", err)
	}

	// Show which oracle decided each boundary.
	slog.SetDefault(slog.New(slog.NewTextHandler(os.Stdout, &slog.HandlerOptions{Level: slog.LevelDebug})))
	DebugMode = true

	tr, err := NewTranscriber(modelsDir, 2, Options{
		FFmpeg: FFmpegConfig{Enabled: true, Timeout: 120 * time.Second},
		Chunk:  ChunkConfig{Enabled: true, Seconds: DefaultChunkSeconds, OverlapSeconds: DefaultChunkOverlapSeconds},
	})
	if err != nil {
		t.Skipf("could not initialize transcriber (needs ONNX Runtime + models): %v", err)
	}
	defer tr.Close()

	data, err := os.ReadFile(audioPath)
	if err != nil {
		t.Fatalf("read audio: %v", err)
	}
	waveform, err := tr.loadAudio(data, "mp3")
	if err != nil {
		t.Fatalf("decode audio (needs ffmpeg): %v", err)
	}
	features := tr.mel.Extract(waveform)
	if len(features) == 0 {
		t.Fatal("no mel features extracted")
	}

	subsampling := int64(tr.config.SubsamplingFactor)
	fps := float64(tr.mel.FramesPerSecond())
	// Absolute encoder frame -> seconds (one encoder frame = subsampling mel frames).
	frameSeconds := float64(subsampling) / fps

	oracle := tr.newBoundaryOracle(features, waveform)
	plan, err := planForAudioWithBoundaries(int64(len(features)), tr.chunkFrames, tr.overlapFrames, subsampling, true, oracle)
	if err != nil {
		t.Fatalf("plan: %v", err)
	}
	t.Logf("planned %d windows over %.1fs of audio", len(plan), float64(len(features))/fps)

	// Decode every window, sharing the exact seam-dedup logic the server uses.
	ctx := context.Background()
	var all []decodedToken
	var prevTail []decodedToken
	for i, win := range plan {
		frameOffset := melToEncoderFrame(win.start, subsampling)
		emitStart := melToEncoderFrame(win.emitStart-win.start, subsampling)
		emitEnd := melToEncoderFrame(win.emitEnd-win.start, subsampling)

		holdFirst := 0
		var resolveSeam func(head []decodedToken) []decodedToken
		if i > 0 {
			holdFirst = seamMaxTokens
			tail := prevTail
			resolveSeam = func(head []decodedToken) []decodedToken { return dedupSeam(tail, head) }
		}

		wt, err := tr.runInference(ctx, features[win.start:win.end], emitStart, emitEnd, frameOffset, holdFirst, resolveSeam, nil)
		if err != nil {
			t.Fatalf("window %d inference: %v", i, err)
		}
		all = append(all, wt...)
		prevTail = wt
	}

	cues := parseSRT(t, srtPath)

	// One report per interior seam.
	for i := 0; i < len(plan)-1; i++ {
		seamSec := float64(plan[i].emitEnd) / fps
		lo := seamSec - seamInspectWindowSeconds
		hi := seamSec + seamInspectWindowSeconds

		t.Logf("========================================================")
		t.Logf("SEAM %d at %s (mel frame %d)", i+1, fmtSeconds(seamSec), plan[i].emitEnd)
		t.Logf("-- transcribed [%s .. %s] --", fmtSeconds(lo), fmtSeconds(hi))
		t.Logf("   %s", tokensTextInRange(tr, all, lo, hi, frameSeconds))
		t.Logf("-- reference SRT [%s .. %s] --", fmtSeconds(lo), fmtSeconds(hi))
		t.Logf("   %s", srtTextInRange(cues, lo, hi))
	}
}

// tokensTextInRange renders the transcribed tokens whose absolute timestep falls
// within [lo, hi] seconds.
func tokensTextInRange(tr *Transcriber, tokens []decodedToken, lo, hi, frameSeconds float64) string {
	var b strings.Builder
	for _, tok := range tokens {
		sec := float64(tok.timestep) * frameSeconds
		if sec < lo || sec > hi {
			continue
		}
		b.WriteString(tr.tokenText(tok.id))
	}
	return strings.TrimSpace(strings.Join(strings.Fields(b.String()), " "))
}

type srtCue struct {
	start, end float64
	text       string
}

// srtTextInRange concatenates the reference cue text overlapping [lo, hi].
func srtTextInRange(cues []srtCue, lo, hi float64) string {
	var parts []string
	for _, c := range cues {
		if c.end < lo || c.start > hi {
			continue
		}
		parts = append(parts, c.text)
	}
	return strings.TrimSpace(strings.Join(parts, " "))
}

// parseSRT reads a SubRip file into time-ordered cues. It is intentionally
// small: enough to line reference text up with seam times, not a full parser.
func parseSRT(t *testing.T, path string) []srtCue {
	f, err := os.Open(path)
	if err != nil {
		t.Logf("no SRT reference (%v); reference column will be empty", err)
		return nil
	}
	defer f.Close()

	var cues []srtCue
	var cur *srtCue
	scanner := bufio.NewScanner(f)
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		switch {
		case line == "":
			if cur != nil {
				cur.text = strings.TrimSpace(cur.text)
				cues = append(cues, *cur)
				cur = nil
			}
		case strings.Contains(line, "-->"):
			parts := strings.Split(line, "-->")
			if len(parts) != 2 {
				continue
			}
			cur = &srtCue{
				start: parseSRTTime(strings.TrimSpace(parts[0])),
				end:   parseSRTTime(strings.TrimSpace(parts[1])),
			}
		case isSRTIndex(line):
			// Skip the numeric index line unless we are mid-cue text.
			if cur != nil {
				cur.text += " " + line
			}
		default:
			if cur != nil {
				cur.text += " " + line
			}
		}
	}
	if cur != nil {
		cur.text = strings.TrimSpace(cur.text)
		cues = append(cues, *cur)
	}
	sort.Slice(cues, func(i, j int) bool { return cues[i].start < cues[j].start })
	return cues
}

func isSRTIndex(line string) bool {
	_, err := strconv.Atoi(line)
	return err == nil
}

// parseSRTTime parses "HH:MM:SS,mmm" into seconds.
func parseSRTTime(s string) float64 {
	s = strings.ReplaceAll(s, ",", ".")
	parts := strings.Split(s, ":")
	if len(parts) != 3 {
		return 0
	}
	h, _ := strconv.ParseFloat(parts[0], 64)
	m, _ := strconv.ParseFloat(parts[1], 64)
	sec, _ := strconv.ParseFloat(parts[2], 64)
	return h*3600 + m*60 + sec
}

func fmtSeconds(sec float64) string {
	if sec < 0 {
		sec = 0
	}
	d := time.Duration(sec * float64(time.Second))
	return d.Truncate(time.Millisecond).String()
}
