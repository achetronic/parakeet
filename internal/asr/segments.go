// SPDX-FileCopyrightText: 2026 Alby Hernández <hola@achetronic.com>
// SPDX-License-Identifier: Apache-2.0

package asr

import "strings"

// This file turns the decoder's per-token timesteps into timed segments.
//
// The TDT decoder already knows WHEN it heard each token: tdtDecode tags every
// emitted token with its absolute encoder-frame timestep (see seam.go) and with
// the frame count the duration head predicted for it. tokensToText throws both
// away and returns a flat string. buildSegments keeps them and groups tokens
// into segments, so the API can report real start/end times instead of one
// synthetic segment spanning the whole file.

// Segment is a run of decoded tokens with wall-clock timing. Start and End are
// seconds from the beginning of the audio.
type Segment struct {
	Start      float64
	End        float64
	Text       string
	Tokens     []int
	AvgLogprob float64
}

const (
	// segmentGapFrames is the silence, in encoder frames (80 ms each), that closes
	// a segment when no sentence punctuation was seen. 8 frames is ~0.64 s: long
	// enough to survive a thinking pause inside a sentence, short enough to catch
	// the pause between two sentences a speaker never punctuates out loud.
	segmentGapFrames int64 = 8

	// segmentMaxTokens bounds one segment's token count, mirroring whisper.cpp's
	// max_seg_len so a punctuation-free monologue still yields readable cues.
	segmentMaxTokens = 220

	// segmentMaxFrames bounds one segment's span. 375 frames is 30 s at the
	// shipped model's 12.5 encoder frames per second.
	segmentMaxFrames int64 = 375
)

// sentenceFinalPunctuation closes a segment when it ends a token's text.
var sentenceFinalPunctuation = []string{".", "!", "?", "…"}

// endsSentence reports whether text ends with sentence-final punctuation,
// ignoring trailing spaces. SentencePiece word-boundary marks are already
// translated to spaces at vocab load, so a token like "▁." arrives as " ." and
// still counts.
func endsSentence(text string) bool {
	trimmed := strings.TrimRight(text, " \t\u00a0")
	if trimmed == "" {
		return false
	}
	for _, p := range sentenceFinalPunctuation {
		if strings.HasSuffix(trimmed, p) {
			return true
		}
	}
	return false
}

// buildSegments groups decoded tokens into timed segments.
//
// tokens must be in decode order and carry ABSOLUTE encoder-frame timesteps: it
// is the same slice tokensToText flattens into the transcript, so the segment
// texts concatenate back to that transcript.
//
// frameSeconds converts an encoder frame to seconds (see
// Transcriber.encoderFrameSeconds; 0.08 s for the shipped model). duration is
// the measured audio length; a segment that would run past it is clamped.
//
// A segment closes before the current token when any of these holds: the gap
// from the previous token's end exceeds segmentGapFrames frames, the previous
// token ended a sentence, the segment already holds segmentMaxTokens tokens, or
// adding this token would span segmentMaxFrames frames. Tokens with no printable
// text (special markers, unknown ids) are dropped, the same rule tokenText
// applies to the transcript.
func buildSegments(tokens []decodedToken, textOf func(int) string, frameSeconds, duration float64) []Segment {
	if len(tokens) == 0 || frameSeconds <= 0 {
		return nil
	}

	segments := make([]Segment, 0, 8)
	var current []decodedToken

	flush := func() {
		if len(current) == 0 {
			return
		}
		seg := segmentFromTokens(current, textOf, frameSeconds)
		if seg.Text != "" {
			if duration > 0 && seg.End > duration {
				seg.End = duration
			}
			if seg.End < seg.Start {
				seg.End = seg.Start
			}
			segments = append(segments, seg)
		}
		current = nil
	}

	for _, tok := range tokens {
		if textOf(tok.id) == "" {
			continue
		}
		if tok.frames < 1 {
			tok.frames = 1
		}

		if len(current) > 0 {
			prev := current[len(current)-1]
			prevEnd := prev.timestep + prev.frames
			switch {
			case tok.timestep-prevEnd > segmentGapFrames:
				flush()
			case endsSentence(textOf(prev.id)):
				flush()
			case len(current) >= segmentMaxTokens:
				flush()
			case tok.timestep+tok.frames-current[0].timestep >= segmentMaxFrames:
				flush()
			}
		}
		current = append(current, tok)
	}
	flush()

	return segments
}

// segmentFromTokens renders one run of tokens into a Segment. Start is the first
// token's timestep, End is the last token's timestep plus the frames the
// duration head predicted for it, so a segment covers the audio its text
// actually came from.
func segmentFromTokens(tokens []decodedToken, textOf func(int) string, frameSeconds float64) Segment {
	if len(tokens) == 0 {
		return Segment{}
	}

	var b strings.Builder
	ids := make([]int, 0, len(tokens))
	var logprobSum float64
	for _, tok := range tokens {
		text := textOf(tok.id)
		if text == "" {
			continue
		}
		b.WriteString(text)
		ids = append(ids, tok.id)
		logprobSum += float64(tok.logprob)
	}
	if len(ids) == 0 {
		return Segment{}
	}

	first, last := tokens[0], tokens[len(tokens)-1]
	lastFrames := last.frames
	if lastFrames < 1 {
		lastFrames = 1
	}

	return Segment{
		Start:      float64(first.timestep) * frameSeconds,
		End:        float64(last.timestep+lastFrames) * frameSeconds,
		Text:       strings.TrimSpace(whitespaceRegex.ReplaceAllString(b.String(), " ")),
		Tokens:     ids,
		AvgLogprob: logprobSum / float64(len(ids)),
	}
}
