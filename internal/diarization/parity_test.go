// SPDX-FileCopyrightText: 2026 Alby Hernández <hola@achetronic.com>
// SPDX-License-Identifier: Apache-2.0

package diarization

import (
	"context"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"testing"
)

// Parity tests: the export from export/diarization against the previously
// shipped conversion (onnx-community/Nemotron-3-Diarization-ONNX). Opt-in,
// both models must be on disk:
//
//	DIAR_REF_DIR=<old> DIAR_NEW_DIR=<new> go test ./internal/diarization/ -run TestParity -v

const (
	// Both graphs quantize independently, so these bound decisions, not bit
	// equality: activity above/below 0.5 and the final turns.
	minCosine        = 0.999
	minDecisionAgree = 0.999 // share of frames where both models agree on active/not
	maxEmbedsDelta   = 2e-1  // absolute, on the 512-d chunk embeddings
	maxLogitsDelta   = 2e0   // sanity cap only, logits saturate far above 0.5
	minArgmaxAgree   = 0.99  // share of active frames with the same dominant speaker
	maxSegmentDrift  = 0.08  // seconds, one mel frame at 16 kHz
)

func parityDirs(t *testing.T) (ref, ours string) {
	t.Helper()
	ref = os.Getenv("DIAR_REF_DIR")
	ours = os.Getenv("DIAR_NEW_DIR")
	if ref == "" || ours == "" {
		t.Skip("DIAR_REF_DIR and DIAR_NEW_DIR not set, skipping export parity tests")
	}
	return ref, ours
}

func loadDiarizer(t *testing.T, dir string) *Diarizer {
	t.Helper()
	d, err := New(dir, 1)
	if err != nil {
		t.Fatalf("New(%q): %v", dir, err)
	}
	t.Cleanup(d.Close)
	return d
}

// TestParityChunkOutputs compares the raw outputs of one streaming step, both
// with an empty speaker cache (first chunk) and with a warm one.
func TestParityChunkOutputs(t *testing.T) {
	refDir, newDir := parityDirs(t)
	ref := loadDiarizer(t, refDir)
	ours := loadDiarizer(t, newDir)

	mel := ref.fe.extract(readWAV(t, "../../testdata/diarization/2-speakers.wav"))
	sub := ref.cfg.SubsamplingFactor
	chunkFrames := ref.cfg.ChunkLength * sub
	if len(mel) < chunkFrames {
		t.Fatalf("audio too short: %d mel frames", len(mel))
	}
	chunk := mel[:chunkFrames]
	chunkEnc := chunkFrames / sub

	t.Run("empty-cache", func(t *testing.T) {
		mask := ones(chunkEnc)
		a := runChunkOutputs(t, "reference", ref, chunk, nil, mask)
		b := runChunkOutputs(t, "exported", ours, chunk, nil, mask)
		compareOutputs(t, "empty-cache", a, b)
	})

	t.Run("warm-cache", func(t *testing.T) {
		_, seed, _, err := ref.runChunk(chunk, nil, ones(chunkEnc))
		if err != nil {
			t.Fatalf("reference runChunk: %v", err)
		}
		cached := seed[:41]
		mask := ones(len(cached) + chunkEnc)
		a := runChunkOutputs(t, "reference", ref, chunk, cached, mask)
		b := runChunkOutputs(t, "exported", ours, chunk, cached, mask)
		compareOutputs(t, "warm-cache", a, b)
	})
}

type chunkOutputs struct {
	logits  []float32
	embeds  []float32
	silence []float32
}

func runChunkOutputs(t *testing.T, name string, d *Diarizer, chunk, cached [][]float32, mask []int64) chunkOutputs {
	t.Helper()
	logits, embeds, silence, err := d.runChunk(chunk, cached, mask)
	if err != nil {
		t.Fatalf("%s runChunk: %v", name, err)
	}
	return chunkOutputs{logits: flatten(logits), embeds: flatten(embeds), silence: silence}
}

func compareOutputs(t *testing.T, label string, a, b chunkOutputs) {
	t.Helper()

	// logits: same speaker decisions (sigmoid > 0.5 <=> logit > 0) and the same
	// shape of the distribution. Absolute deltas stay a diagnostic because a
	// saturated logit can move without changing any decision.
	if agree := decisionAgreement(a.logits, b.logits); agree < minDecisionAgree {
		t.Fatalf("%s logits: models agree on %.4f%% of frames, want >= %.1f%%",
			label, 100*agree, 100*minDecisionAgree)
	}
	if cosine := cosineSimilarity(a.logits, b.logits); cosine < minCosine {
		t.Fatalf("%s logits: cosine %.6f < %.6f", label, cosine, minCosine)
	}
	if delta := maxAbs(a.logits, b.logits); delta > maxLogitsDelta {
		t.Fatalf("%s logits: max abs delta %.3f > %.3f", label, delta, maxLogitsDelta)
	}

	// chunk_embeds feeds the next step's cache, so bound the absolute drift.
	checkSimilar(t, label+" chunk_embeds", a.embeds, b.embeds, maxEmbedsDelta, minCosine)

	if delta := maxAbs(a.silence, b.silence); delta > 1e-5 {
		t.Fatalf("%s: silence_embeds differ, max abs %g", label, delta)
	}

	t.Logf("%s: logits delta %.3f cosine %.7f, decisions agree %.4f%%, embeds delta %.4f, silence_embeds identical",
		label, maxAbs(a.logits, b.logits), cosineSimilarity(a.logits, b.logits),
		100*decisionAgreement(a.logits, b.logits), maxAbs(a.embeds, b.embeds))
}

// decisionAgreement is the share of frames where both models put the same
// speakers above the 0.5 activity threshold (logits and sigmoid cross zero at
// the same point).
func decisionAgreement(a, b []float32) float64 {
	if len(a) != len(b) || len(a) == 0 {
		return 0
	}
	same := 0
	for i := range a {
		if (a[i] > 0) == (b[i] > 0) {
			same++
		}
	}
	return float64(same) / float64(len(a))
}

// TestParityWholeFile compares per-frame speaker activity over whole files,
// through the real chunking and speaker-cache policy.
func TestParityWholeFile(t *testing.T) {
	refDir, newDir := parityDirs(t)
	ref := loadDiarizer(t, refDir)
	ours := loadDiarizer(t, newDir)

	for _, audio := range parityAudio(t) {
		t.Run(filepath.Base(audio), func(t *testing.T) {
			samples := readWAV(t, audio)
			a, err := ref.run(context.Background(), samples)
			if err != nil {
				t.Fatalf("reference run: %v", err)
			}
			b, err := ours.run(context.Background(), samples)
			if err != nil {
				t.Fatalf("exported run: %v", err)
			}
			if len(a) != len(b) {
				t.Fatalf("frame count differs: reference %d, exported %d", len(a), len(b))
			}

			pa, pb := sigmoidRows(a), sigmoidRows(b)
			fa, fb := flatten(pa), flatten(pb)
			checkSimilar(t, "activity", fa, fb, maxLogitsDelta, minCosine)

			agree, active := argmaxAgreement(pa, pb)
			ratio := float64(agree) / float64(active)
			if ratio < minArgmaxAgree {
				t.Fatalf("dominant speaker agrees on %.2f%% of %d active frames, want >= %.0f%%",
					100*ratio, active, 100*minArgmaxAgree)
			}
			t.Logf("%.1fs audio, %d frames (%d active): activity delta %.4f, speaker agrees on %.2f%%",
				float64(len(samples))/float64(ref.cfg.SampleRate), len(a), active,
				maxAbs(fa, fb), 100*ratio)
		})
	}
}

// TestParitySegments compares the final speaker turns.
func TestParitySegments(t *testing.T) {
	refDir, newDir := parityDirs(t)
	ref := loadDiarizer(t, refDir)
	ours := loadDiarizer(t, newDir)

	for _, audio := range parityAudio(t) {
		t.Run(filepath.Base(audio), func(t *testing.T) {
			samples := readWAV(t, audio)
			a, err := ref.Diarize(context.Background(), samples)
			if err != nil {
				t.Fatalf("reference Diarize: %v", err)
			}
			b, err := ours.Diarize(context.Background(), samples)
			if err != nil {
				t.Fatalf("exported Diarize: %v", err)
			}
			if len(a) != len(b) {
				t.Fatalf("segment count differs: reference %d, exported %d\nref=%v\nours=%v",
					len(a), len(b), a, b)
			}
			for i := range a {
				if a[i].Speaker != b[i].Speaker {
					t.Fatalf("segment %d speaker differs: reference %d, exported %d",
						i, a[i].Speaker, b[i].Speaker)
				}
				drift := math.Max(math.Abs(a[i].Start-b[i].Start), math.Abs(a[i].End-b[i].End))
				if drift > maxSegmentDrift {
					t.Fatalf("segment %d drifts %.3fs: reference %+v, exported %+v", i, drift, a[i], b[i])
				}
			}
			t.Logf("%d segments: same speakers, boundaries within %.2fs", len(a), maxSegmentDrift)
		})
	}
}

// parityAudio returns the bundled audio used for the comparisons: the
// two-speaker sample plus the first five minutes of the long reference clip,
// decoded to 16 kHz mono WAV in a temp dir. Five minutes is enough chunks
// (~11) for the speaker cache to fill up and compress, which is the path that
// compounds quantization error the most.
func parityAudio(t *testing.T) []string {
	t.Helper()
	audios := []string{"../../testdata/diarization/2-speakers.wav"}

	src := "../../testdata/reference/learn-case-interviews.mp3"
	if _, err := os.Stat(src); err != nil {
		return audios
	}
	if _, err := exec.LookPath("ffmpeg"); err != nil {
		t.Logf("ffmpeg not found, skipping %s", src)
		return audios
	}
	out := filepath.Join(t.TempDir(), "reference-5min.wav")
	cmd := exec.Command("ffmpeg", "-v", "error", "-i", src, "-t", "300", "-ar", "16000", "-ac", "1", "-y", out)
	if err := cmd.Run(); err != nil {
		t.Fatalf("ffmpeg: %v", err)
	}
	return append(audios, out)
}

func ones(n int) []int64 {
	out := make([]int64, n)
	for i := range out {
		out[i] = 1
	}
	return out
}

func flatten(rows [][]float32) []float32 {
	if len(rows) == 0 {
		return nil
	}
	out := make([]float32, 0, len(rows)*len(rows[0]))
	for _, r := range rows {
		out = append(out, r...)
	}
	return out
}

func maxAbs(a, b []float32) float32 {
	if len(a) != len(b) {
		return float32(math.Inf(1))
	}
	worst := float32(0)
	for i := range a {
		if d := float32(math.Abs(float64(a[i] - b[i]))); d > worst {
			worst = d
		}
	}
	return worst
}

// checkSimilar fails when two vectors drift more than the given bounds.
func checkSimilar(t *testing.T, label string, a, b []float32, maxDelta, minCosine float64) {
	t.Helper()
	if len(a) != len(b) {
		t.Fatalf("%s: length differs: %d != %d", label, len(a), len(b))
	}
	if delta := float64(maxAbs(a, b)); delta > maxDelta {
		t.Fatalf("%s: max abs delta %.4f > %.4f", label, delta, maxDelta)
	}
	if cosine := cosineSimilarity(a, b); cosine < minCosine {
		t.Fatalf("%s: cosine %.6f < %.6f", label, cosine, minCosine)
	}
}

func cosineSimilarity(a, b []float32) float64 {
	var dot, na, nb float64
	for i := range a {
		dot += float64(a[i]) * float64(b[i])
		na += float64(a[i]) * float64(a[i])
		nb += float64(b[i]) * float64(b[i])
	}
	if na == 0 || nb == 0 {
		return 0
	}
	return dot / (math.Sqrt(na) * math.Sqrt(nb))
}

func sigmoidRows(logits [][]float32) [][]float32 {
	out := make([][]float32, len(logits))
	for f := range logits {
		out[f] = make([]float32, len(logits[f]))
		for s, v := range logits[f] {
			out[f][s] = float32(1 / (1 + math.Exp(-float64(v))))
		}
	}
	return out
}

// argmaxAgreement counts frames where both models pick the same dominant
// speaker, ignoring frames where neither is active.
func argmaxAgreement(a, b [][]float32) (agree, active int) {
	for f := range a {
		ia, va := argmax(a[f])
		ib, vb := argmax(b[f])
		if va < 0.5 && vb < 0.5 {
			continue
		}
		active++
		if ia == ib {
			agree++
		}
	}
	return agree, active
}

func argmax(row []float32) (int, float32) {
	best, arg := float32(-1), -1
	for i, v := range row {
		if v > best {
			best, arg = v, i
		}
	}
	return arg, best
}
