// SPDX-FileCopyrightText: 2026 Alby Hernández <hola@achetronic.com>
// SPDX-License-Identifier: Apache-2.0

package server

import (
	"encoding/json"
	"math"
	"strings"
	"testing"

	"parakeet/internal/asr"
)

func TestToAPISegments(t *testing.T) {
	// 803 encoder frames * 0.08 s is 64.24 s, which lands just below 64240 ms in
	// binary floating point. Truncating the product would report 64239.
	in := []asr.Segment{
		{Start: 0.8, End: 1.2, Text: "Hello there", Tokens: []int{1, 2}, AvgLogprob: -0.2},
		{Start: 2.4, End: 2.8, Text: "Second one", Tokens: nil, AvgLogprob: -0.6},
		{Start: float64(803) * 0.08, End: 65.0, Text: "Frame drift", Tokens: []int{7}, AvgLogprob: -0.3},
	}

	out := toAPISegments(in)
	if len(out) != len(in) {
		t.Fatalf("got %d segments, want %d", len(out), len(in))
	}

	for i, seg := range out {
		if seg.ID != i {
			t.Errorf("segment %d: id = %d, want %d", i, seg.ID, i)
		}
		if seg.Seek != int(math.Round(in[i].Start*1000)) {
			t.Errorf("segment %d: seek = %d, want %d ms", i, seg.Seek, int(math.Round(in[i].Start*1000)))
		}
		if seg.Start != in[i].Start || seg.End != in[i].End {
			t.Errorf("segment %d: timing = %v-%v, want %v-%v", i, seg.Start, seg.End, in[i].Start, in[i].End)
		}
		if seg.Tokens == nil {
			t.Errorf("segment %d: tokens is nil, want an empty array so the JSON key is present", i)
		}
		if seg.AvgLogprob != in[i].AvgLogprob {
			t.Errorf("segment %d: avg_logprob = %v, want %v", i, seg.AvgLogprob, in[i].AvgLogprob)
		}
		if seg.CompressionRatio <= 0 {
			t.Errorf("segment %d: compression_ratio = %v, want a positive measured value", i, seg.CompressionRatio)
		}
	}

	if out[2].Seek != 64240 {
		t.Errorf("seek = %d, want 64240 ms", out[2].Seek)
	}
	if out[0].Text != "Hello there" || out[1].Text != "Second one" {
		t.Errorf("texts = %q, %q", out[0].Text, out[1].Text)
	}
}

func TestVerboseJSONAlwaysCarriesSegments(t *testing.T) {
	// A silent file produces no segments; the key must still be present so
	// clients parsing the OpenAI schema do not trip on a missing field.
	body, err := json.Marshal(VerboseTranscriptionResponse{
		Task:     "transcribe",
		Language: "en",
		Duration: 1.0,
		Text:     "",
		Segments: toAPISegments(nil),
	})
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	if !strings.Contains(string(body), `"segments":[]`) {
		t.Errorf("body = %s, want an empty segments array", body)
	}
}

func TestRenderSRT(t *testing.T) {
	segments := []asr.Segment{
		{Start: 0.8, End: 1.2, Text: "Hello there"},
		{Start: 2.4, End: 2.8, Text: "Second one"},
	}

	want := "1\n00:00:00,800 --> 00:00:01,200\nHello there\n\n" +
		"2\n00:00:02,400 --> 00:00:02,800\nSecond one\n\n"
	if got := renderSRT(segments); got != want {
		t.Errorf("renderSRT =\n%q\nwant\n%q", got, want)
	}
	if got := renderSRT(nil); got != "" {
		t.Errorf("renderSRT(nil) = %q, want empty", got)
	}
}

func TestRenderVTT(t *testing.T) {
	segments := []asr.Segment{{Start: 0.8, End: 1.2, Text: "Hello there"}}

	want := "WEBVTT\n\n00:00:00.800 --> 00:00:01.200\nHello there\n\n"
	if got := renderVTT(segments); got != want {
		t.Errorf("renderVTT =\n%q\nwant\n%q", got, want)
	}
	if got := renderVTT(nil); got != "WEBVTT\n\n" {
		t.Errorf("renderVTT(nil) = %q, want just the header", got)
	}
}

func TestFormatTimestamps(t *testing.T) {
	tests := []struct {
		seconds float64
		srt     string
		vtt     string
	}{
		// 1.2 s must not render as 199 ms: 1.2-1 is 0.19999999999999998 in
		// binary floating point, which truncation turns into 199.
		{1.2, "00:00:01,200", "00:00:01.200"},
		{0.8, "00:00:00,800", "00:00:00.800"},
		// Rounding once at the millisecond level carries across field borders.
		{59.9996, "00:01:00,000", "00:01:00.000"},
		{3599.9999, "01:00:00,000", "01:00:00.000"},
		{0, "00:00:00,000", "00:00:00.000"},
		{-1, "00:00:00,000", "00:00:00.000"},
	}
	for _, tc := range tests {
		if got := formatSRTTime(tc.seconds); got != tc.srt {
			t.Errorf("formatSRTTime(%v) = %q, want %q", tc.seconds, got, tc.srt)
		}
		if got := formatVTTTime(tc.seconds); got != tc.vtt {
			t.Errorf("formatVTTTime(%v) = %q, want %q", tc.seconds, got, tc.vtt)
		}
	}
}

func TestCompressionRatio(t *testing.T) {
	if got := compressionRatio(""); !approx(got, 1.0) {
		t.Errorf("compressionRatio(\"\") = %v, want 1.0", got)
	}

	// Highly redundant text compresses well, so the ratio must be well above 1.
	if got := compressionRatio(strings.Repeat("a", 1000)); got <= 2 {
		t.Errorf("compressionRatio of 1000 repeated chars = %v, want > 2", got)
	}
}

func approx(a, b float64) bool {
	return a > b-1e-9 && a < b+1e-9
}
