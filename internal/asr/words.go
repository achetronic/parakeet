// SPDX-FileCopyrightText: 2026 Alby Hernández <hola@achetronic.com>
// SPDX-License-Identifier: Apache-2.0

package asr

import (
	"context"
	"strings"
)

// Word is a single whitespace-delimited word with its time span in seconds.
// Start is the first token's encoder-frame timestep; End is the last token's
// timestep plus its TDT duration.
type Word struct {
	Text  string
	Start float64
	End   float64
}

// TranscribeWords transcribes audio and returns each word with its start and
// end times. Word boundaries follow the SentencePiece word-boundary marker (a
// leading space in the vocab entry); punctuation and continuation tokens attach
// to the preceding word.
func (t *Transcriber) TranscribeWords(ctx context.Context, audioData []byte, format, language string) ([]Word, error) {
	tokens, err := t.transcribe(ctx, audioData, format, language, nil)
	if err != nil {
		return nil, err
	}
	return t.wordsFromTokens(tokens), nil
}

// TranscribeWordsSamples transcribes already-loaded mono 16 kHz samples and
// returns each word with its start and end times.
func (t *Transcriber) TranscribeWordsSamples(ctx context.Context, samples []float32) ([]Word, error) {
	tokens, err := t.transcribeSamples(ctx, samples, nil)
	if err != nil {
		return nil, err
	}
	return t.wordsFromTokens(tokens), nil
}

// wordsFromTokens groups decoded tokens into words with timestamps. A token
// whose vocab text starts with a space opens a new word; every other token is
// appended to the current word, extending its end time.
func (t *Transcriber) wordsFromTokens(tokens []decodedToken) []Word {
	// One encoder frame spans subsampling mel frames: subsampling * hop / sampleRate seconds.
	frameSec := float64(t.config.SubsamplingFactor) * float64(t.mel.HopLength()) / 16000.0

	var words []Word
	for _, tok := range tokens {
		text := t.tokenText(tok.id)
		if text == "" {
			continue
		}

		start := float64(tok.timestep) * frameSec
		dur := tok.duration
		if dur < 1 {
			dur = 1
		}
		end := float64(tok.timestep+dur) * frameSec

		if strings.HasPrefix(text, " ") {
			words = append(words, Word{Text: strings.TrimSpace(text), Start: start, End: end})
			continue
		}
		if len(words) == 0 {
			words = append(words, Word{Text: text, Start: start, End: end})
			continue
		}
		last := &words[len(words)-1]
		last.Text += text
		if end > last.End {
			last.End = end
		}
	}
	return words
}
