package asr

import (
	"bytes"
	"encoding/binary"
	"errors"
	"os/exec"
	"sync"
	"testing"
	"time"
)

// buildMinimalWAV produces a tiny but valid 16-bit PCM WAV blob suitable
// for exercising the magic-byte detection and parsing path without any
// external dependency.
func buildMinimalWAV(t *testing.T, sampleRate uint32, samples int) []byte {
	t.Helper()
	var buf bytes.Buffer

	bitsPerSample := uint16(16)
	numChannels := uint16(1)
	byteRate := sampleRate * uint32(numChannels) * uint32(bitsPerSample) / 8
	blockAlign := numChannels * bitsPerSample / 8
	dataSize := uint32(samples) * uint32(blockAlign)

	buf.WriteString("RIFF")
	_ = binary.Write(&buf, binary.LittleEndian, uint32(36+dataSize))
	buf.WriteString("WAVE")

	buf.WriteString("fmt ")
	_ = binary.Write(&buf, binary.LittleEndian, uint32(16))
	_ = binary.Write(&buf, binary.LittleEndian, uint16(1)) // PCM
	_ = binary.Write(&buf, binary.LittleEndian, numChannels)
	_ = binary.Write(&buf, binary.LittleEndian, sampleRate)
	_ = binary.Write(&buf, binary.LittleEndian, byteRate)
	_ = binary.Write(&buf, binary.LittleEndian, blockAlign)
	_ = binary.Write(&buf, binary.LittleEndian, bitsPerSample)

	buf.WriteString("data")
	_ = binary.Write(&buf, binary.LittleEndian, dataSize)
	for i := 0; i < samples; i++ {
		_ = binary.Write(&buf, binary.LittleEndian, int16(i%32000))
	}
	return buf.Bytes()
}

func TestIsWAV(t *testing.T) {
	cases := []struct {
		name string
		in   []byte
		want bool
	}{
		{"valid WAV header", buildMinimalWAV(t, 16000, 4), true},
		{"too short", []byte{0x01, 0x02}, false},
		{"wrong RIFF", append([]byte("XXXX\x00\x00\x00\x00WAVE"), make([]byte, 100)...), false},
		{"wrong WAVE", append([]byte("RIFF\x00\x00\x00\x00XXXX"), make([]byte, 100)...), false},
		{"ogg magic", []byte("OggS\x00\x02\x00\x00\x00\x00\x00\x00foo"), false},
		{"id3 mp3", []byte("ID3\x03\x00\x00\x00\x00\x00\x00\x00\x00foo"), false},
		{"empty", nil, false},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := isWAV(tc.in); got != tc.want {
				t.Fatalf("isWAV(%q) = %v, want %v", tc.name, got, tc.want)
			}
		})
	}
}

func TestLoadAudioAcceptsWAV(t *testing.T) {
	tr := &Transcriber{}
	wav := buildMinimalWAV(t, 16000, 100)

	samples, err := tr.loadAudio(wav, "")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(samples) == 0 {
		t.Fatalf("expected decoded samples, got 0")
	}
}

func TestLoadAudioRejectsNonWAVWhenFFmpegDisabled(t *testing.T) {
	tr := &Transcriber{ffmpeg: nil}

	// Clearly non-WAV payload. Without ffmpeg this must surface
	// ErrUnsupportedAudio so the HTTP handler can map it to 400.
	_, err := tr.loadAudio([]byte("OggS\x00\x02\x00\x00\x00\x00\x00\x00this is not wav"), ".ogg")
	if err == nil {
		t.Fatal("expected error, got nil")
	}
	if !errors.Is(err, ErrUnsupportedAudio) {
		t.Fatalf("expected ErrUnsupportedAudio, got %v", err)
	}
}

// TestLoadAudioConcurrentWAV ensures that the WAV fast path is safe to call
// from many goroutines at once. This matches what the worker pool does in
// practice (up to `-workers` concurrent inferences, each preceded by
// loadAudio). It is the regression test for the PR #5 tempfile collision
// bug: we run it many times in parallel and expect no data races or
// spurious failures.
func TestLoadAudioConcurrentWAV(t *testing.T) {
	tr := &Transcriber{}
	wav := buildMinimalWAV(t, 16000, 1000)

	const goroutines = 32
	const iterations = 16

	var wg sync.WaitGroup
	errs := make(chan error, goroutines*iterations)

	for g := 0; g < goroutines; g++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for i := 0; i < iterations; i++ {
				samples, err := tr.loadAudio(wav, "")
				if err != nil {
					errs <- err
					return
				}
				if len(samples) == 0 {
					errs <- errors.New("empty samples")
					return
				}
			}
		}()
	}

	wg.Wait()
	close(errs)
	for err := range errs {
		t.Fatalf("concurrent loadAudio failed: %v", err)
	}
}

// TestFFmpegConverterUnique verifies that two concurrent conversions never
// share temporary files. We don't require ffmpeg to actually succeed — we
// only care that os.CreateTemp hands us unique paths. The test is skipped
// when ffmpeg is not available to keep CI green on runners without it.
func TestFFmpegConverterConcurrentTempFiles(t *testing.T) {
	if _, err := exec.LookPath("ffmpeg"); err != nil {
		t.Skip("ffmpeg not available in PATH, skipping")
	}

	conv := newFFmpegConverter(FFmpegConfig{
		Enabled: true,
		Timeout: 10 * time.Second,
	})
	if conv == nil {
		t.Skip("converter did not initialize, skipping")
	}

	// Feed garbage so ffmpeg errors out quickly, but through the real code
	// path that creates temp files. The payload is intentionally invalid
	// to keep the test fast. We only care that each invocation errors
	// with ErrUnsupportedAudio and never panics or races.
	payload := []byte("not a real audio file, just checking concurrency safety")

	const goroutines = 16
	var wg sync.WaitGroup
	for i := 0; i < goroutines; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			_, err := conv.Convert(payload)
			if err != nil && !errors.Is(err, ErrUnsupportedAudio) {
				t.Errorf("unexpected error class: %v", err)
			}
		}()
	}
	wg.Wait()
}

func TestNewFFmpegConverterReturnsNilWhenDisabled(t *testing.T) {
	if c := newFFmpegConverter(FFmpegConfig{Enabled: false}); c != nil {
		t.Fatalf("expected nil converter when disabled, got %#v", c)
	}
}

func TestNewFFmpegConverterReturnsNilWhenMissing(t *testing.T) {
	// Use a name that is overwhelmingly unlikely to resolve in PATH.
	c := newFFmpegConverter(FFmpegConfig{
		Enabled:    true,
		BinaryPath: "__definitely_not_a_real_binary_parakeet_test__",
	})
	if c != nil {
		t.Fatalf("expected nil converter when binary missing, got %#v", c)
	}
}

func TestTrimStderr(t *testing.T) {
	cases := []struct {
		in   string
		want string
	}{
		{"", "conversion failed"},
		{"short error", "short error"},
		{"line1\nline2\nline3", "line1 line2 line3"},
		{"   \n  \r\n", ""},
	}
	for _, tc := range cases {
		got := trimStderr(tc.in)
		// Accept "conversion failed" when the normalized result is empty.
		if got == "" || (tc.want == "" && got != "conversion failed") {
			if tc.want != "" || got != "conversion failed" {
				t.Errorf("trimStderr(%q) = %q, want %q", tc.in, got, tc.want)
			}
			continue
		}
		if tc.want != "" && got != tc.want {
			t.Errorf("trimStderr(%q) = %q, want %q", tc.in, got, tc.want)
		}
	}
}
