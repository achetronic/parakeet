// SPDX-FileCopyrightText: 2026 Alby Hernández <hola@achetronic.com>
// SPDX-License-Identifier: Apache-2.0

package asr

import (
	"math"
	"strings"
	"testing"
)

// textOf stands in for the vocab lookup tokenText performs at runtime.
func textOf(texts map[int]string) func(int) string {
	return func(id int) string { return texts[id] }
}

// frameSeconds is the shipped model's encoder frame length: subsampling 8 *
// mel hop 160 samples / 16000 Hz.
const testFrameSeconds = 0.08

func approxEqual(a, b float64) bool {
	return math.Abs(a-b) < 1e-6
}

func TestEndsSentence(t *testing.T) {
	tests := []struct {
		text string
		want bool
	}{
		{"Hello there.", true},
		{" stop!", true},
		{" really?", true},
		{" trailing ellipsis…", true},
		{" period with trailing space. ", true},
		{" no punctuation", false},
		{"", false},
		{"   ", false},
		{"3.5", false},
	}
	for _, tc := range tests {
		if got := endsSentence(tc.text); got != tc.want {
			t.Errorf("endsSentence(%q) = %v; want %v", tc.text, got, tc.want)
		}
	}
}

func TestBuildSegments_SplitsOnSilenceGap(t *testing.T) {
	texts := map[int]string{1: "Hello", 2: " there", 3: " Second", 4: " one"}
	tokens := []decodedToken{
		{id: 1, timestep: 10, frames: 2, logprob: -0.1},
		{id: 2, timestep: 13, frames: 2, logprob: -0.3},
		// 30 - (13+2) = 15 frames of silence, past segmentGapFrames.
		{id: 3, timestep: 30, frames: 2, logprob: -0.5},
		{id: 4, timestep: 33, frames: 2, logprob: -0.7},
	}

	segments := buildSegments(tokens, textOf(texts), testFrameSeconds, 10)
	if len(segments) != 2 {
		t.Fatalf("got %d segments, want 2: %+v", len(segments), segments)
	}

	want := []Segment{
		{Start: 0.8, End: 1.2, Text: "Hello there", Tokens: []int{1, 2}, AvgLogprob: -0.2},
		{Start: 2.4, End: 2.8, Text: "Second one", Tokens: []int{3, 4}, AvgLogprob: -0.6},
	}
	assertSegments(t, segments, want)
}

func TestBuildSegments_SplitsOnPunctuation(t *testing.T) {
	texts := map[int]string{1: "One.", 2: " Two", 3: " three.", 4: " Four"}
	tokens := []decodedToken{
		{id: 1, timestep: 0, frames: 1},
		{id: 2, timestep: 2, frames: 1},
		{id: 3, timestep: 4, frames: 1},
		{id: 4, timestep: 6, frames: 1},
	}

	segments := buildSegments(tokens, textOf(texts), testFrameSeconds, 10)
	if len(segments) != 3 {
		t.Fatalf("got %d segments, want 3: %+v", len(segments), segments)
	}

	want := []Segment{
		{Start: 0.00, End: 0.08, Text: "One.", Tokens: []int{1}},
		{Start: 0.16, End: 0.40, Text: "Two three.", Tokens: []int{2, 3}},
		{Start: 0.48, End: 0.56, Text: "Four", Tokens: []int{4}},
	}
	assertSegments(t, segments, want)
}

func TestBuildSegments_CapsSegmentTokenCount(t *testing.T) {
	const total = 250
	texts := make(map[int]string, total)
	tokens := make([]decodedToken, 0, total)
	for i := 0; i < total; i++ {
		id := i + 1
		texts[id] = " word"
		// One frame per token: 0.08 s each, so 250 tokens span 20 s and the
		// token cap binds before the 30 s span cap.
		tokens = append(tokens, decodedToken{id: id, timestep: int64(i), frames: 1})
	}

	segments := buildSegments(tokens, textOf(texts), testFrameSeconds, 1e6)
	if len(segments) != 2 {
		t.Fatalf("got %d segments, want 2 (220 + 30 tokens)", len(segments))
	}
	if len(segments[0].Tokens) != segmentMaxTokens {
		t.Errorf("first segment has %d tokens, want the %d cap", len(segments[0].Tokens), segmentMaxTokens)
	}
	if len(segments[1].Tokens) != total-segmentMaxTokens {
		t.Errorf("second segment has %d tokens, want %d", len(segments[1].Tokens), total-segmentMaxTokens)
	}
}

func TestBuildSegments_CapsSegmentSpan(t *testing.T) {
	const total = 400
	texts := make(map[int]string, total)
	tokens := make([]decodedToken, 0, total)
	for i := 0; i < total; i++ {
		id := i + 1
		texts[id] = " word"
		// Two frames per token: 0.16 s each, so the 30 s span cap binds at 187
		// tokens, before the 220 token cap.
		tokens = append(tokens, decodedToken{id: id, timestep: int64(i) * 2, frames: 1})
	}

	segments := buildSegments(tokens, textOf(texts), testFrameSeconds, 1e6)
	if len(segments) < 2 {
		t.Fatalf("got %d segments, want at least 2", len(segments))
	}
	span := segments[0].End - segments[0].Start
	if span > 30.0 {
		t.Errorf("first segment spans %v s, want at most 30 s", span)
	}
	if len(segments[0].Tokens) >= segmentMaxTokens {
		t.Errorf("first segment has %d tokens, want the span cap to bind first", len(segments[0].Tokens))
	}
}

func TestBuildSegments_DropsUnprintableTokens(t *testing.T) {
	texts := map[int]string{1: "Real", 2: " words"}
	// id 99 is absent from the vocab map, standing in for a special or unknown
	// token id: tokenText returns "" for it and it must not reach a segment.
	tokens := []decodedToken{
		{id: 99, timestep: 0, frames: 1},
		{id: 1, timestep: 5, frames: 1},
		{id: 99, timestep: 6, frames: 1},
		{id: 2, timestep: 7, frames: 1},
	}

	segments := buildSegments(tokens, textOf(texts), testFrameSeconds, 10)
	if len(segments) != 1 {
		t.Fatalf("got %d segments, want 1: %+v", len(segments), segments)
	}
	if segments[0].Text != "Real words" {
		t.Errorf("text = %q, want %q", segments[0].Text, "Real words")
	}
	if got := segments[0].Tokens; len(got) != 2 || got[0] != 1 || got[1] != 2 {
		t.Errorf("tokens = %v, want [1 2]", got)
	}
	// Timing follows the printable tokens, not the dropped ones.
	if !approxEqual(segments[0].Start, 5*testFrameSeconds) {
		t.Errorf("start = %v, want %v", segments[0].Start, 5*testFrameSeconds)
	}
}

func TestBuildSegments_ClampsEndToDuration(t *testing.T) {
	texts := map[int]string{1: "Tail"}
	// Start is 4.8 s and the token's predicted span would run to 5.6 s in a file
	// that only measures 5 s.
	tokens := []decodedToken{{id: 1, timestep: 60, frames: 10}}

	segments := buildSegments(tokens, textOf(texts), testFrameSeconds, 5)
	if len(segments) != 1 {
		t.Fatalf("got %d segments, want 1", len(segments))
	}
	if !approxEqual(segments[0].Start, 4.8) {
		t.Errorf("start = %v, want 4.8", segments[0].Start)
	}
	if !approxEqual(segments[0].End, 5) {
		t.Errorf("end = %v, want clamped to 5", segments[0].End)
	}
}

func TestBuildSegments_DegenerateInput(t *testing.T) {
	texts := map[int]string{1: "x"}
	tokens := []decodedToken{{id: 1, timestep: 0, frames: 1}}

	if got := buildSegments(nil, textOf(texts), testFrameSeconds, 10); got != nil {
		t.Errorf("no tokens: got %v, want nil", got)
	}
	if got := buildSegments(tokens, textOf(texts), 0, 10); got != nil {
		t.Errorf("zero frame rate: got %v, want nil", got)
	}
	if got := buildSegments(tokens, textOf(map[int]string{}), testFrameSeconds, 10); len(got) != 0 {
		t.Errorf("no printable text: got %v, want no segments", got)
	}
}

func TestBuildSegments_SegmentTextsRebuildTranscript(t *testing.T) {
	// The segments and the transcript come from the same token list, so joining
	// the segment texts must reproduce tokensToText's output.
	texts := map[int]string{1: "One.", 2: " Two", 3: " three.", 4: " Four"}
	tokens := []decodedToken{
		{id: 1, timestep: 0, frames: 1},
		{id: 2, timestep: 2, frames: 1},
		{id: 3, timestep: 4, frames: 1},
		{id: 4, timestep: 6, frames: 1},
	}

	transcriber := &Transcriber{vocab: texts}
	transcript := transcriber.tokensToText(tokens)

	joined := ""
	for _, seg := range buildSegments(tokens, transcriber.tokenText, testFrameSeconds, 10) {
		joined += " " + seg.Text
	}
	if got := strings.TrimSpace(joined); got != transcript {
		t.Errorf("joined segments = %q; transcript = %q", got, transcript)
	}
}

func TestEncoderFrameSeconds(t *testing.T) {
	// The shipped model: 128 mel bins at 16 kHz with a 160-sample hop and 8x
	// Conformer subsampling, so one encoder frame is 0.08 s (12.5 per second).
	transcriber := &Transcriber{
		config: Config{FeaturesSize: 128, SubsamplingFactor: 8},
		mel:    NewMelFilterbank(128, 16000),
	}
	if got := transcriber.encoderFrameSeconds(); !approxEqual(got, 0.08) {
		t.Errorf("encoderFrameSeconds = %v, want 0.08", got)
	}

	// An unusable subsampling factor reports "no timing available" rather than
	// inventing a rate, and buildSegments then returns no segments.
	broken := &Transcriber{
		config: Config{FeaturesSize: 128, SubsamplingFactor: 0},
		mel:    NewMelFilterbank(128, 16000),
	}
	if got := broken.encoderFrameSeconds(); got != 0 {
		t.Errorf("encoderFrameSeconds with subsampling 0 = %v, want 0", got)
	}
	tokens := []decodedToken{{id: 1, timestep: 0, frames: 1}}
	textOfOne := func(int) string { return "x" }
	if got := buildSegments(tokens, textOfOne, broken.encoderFrameSeconds(), 10); got != nil {
		t.Errorf("buildSegments with no timing = %v, want nil", got)
	}
}

func assertSegments(t *testing.T, got, want []Segment) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("got %d segments, want %d", len(got), len(want))
	}
	for i := range want {
		g, w := got[i], want[i]
		if !approxEqual(g.Start, w.Start) {
			t.Errorf("segment %d: start = %v, want %v", i, g.Start, w.Start)
		}
		if !approxEqual(g.End, w.End) {
			t.Errorf("segment %d: end = %v, want %v", i, g.End, w.End)
		}
		if g.Text != w.Text {
			t.Errorf("segment %d: text = %q, want %q", i, g.Text, w.Text)
		}
		if len(g.Tokens) != len(w.Tokens) {
			t.Errorf("segment %d: tokens = %v, want %v", i, g.Tokens, w.Tokens)
			continue
		}
		for j := range w.Tokens {
			if g.Tokens[j] != w.Tokens[j] {
				t.Errorf("segment %d: tokens = %v, want %v", i, g.Tokens, w.Tokens)
				break
			}
		}
		if !approxEqual(g.AvgLogprob, w.AvgLogprob) {
			t.Errorf("segment %d: avg_logprob = %v, want %v", i, g.AvgLogprob, w.AvgLogprob)
		}
	}
}

func TestLogProb(t *testing.T) {
	t.Run("log-probability inputs pass through", func(t *testing.T) {
		// ln([0.5, 0.25, 0.25]): logsumexp is 0, so the value is returned as is.
		logits := []float32{-0.6931472, -1.3862944, -1.3862944}
		if got := logProb(logits, 0); !approxEqual(float64(got), -0.6931472) {
			t.Errorf("logProb = %v, want -0.6931472", got)
		}
	})

	t.Run("raw logits are normalised", func(t *testing.T) {
		// softmax([2,1,0])[0] = e^2 / (e^2+e+1) = 0.665241, ln of that is -0.407605.
		logits := []float32{2, 1, 0}
		if got := logProb(logits, 0); !approxEqual(float64(got), -0.4076051) {
			t.Errorf("logProb = %v, want -0.4076051", got)
		}
	})

	t.Run("out of range and empty inputs return 0", func(t *testing.T) {
		if got := logProb([]float32{1, 2}, 5); got != 0 {
			t.Errorf("logProb out of range = %v, want 0", got)
		}
		if got := logProb(nil, 0); got != 0 {
			t.Errorf("logProb empty = %v, want 0", got)
		}
	})
}
