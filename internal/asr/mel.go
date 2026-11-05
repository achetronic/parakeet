// SPDX-FileCopyrightText: 2026 Alby Hernández <hola@achetronic.com>
// SPDX-License-Identifier: Apache-2.0

package asr

import (
	"log/slog"
	"math"
	"math/cmplx"
)

// NeMo AudioToMelSpectrogramPreprocessor constants. The extractor below
// mirrors the frontend the model was trained on: a 0.97 pre-emphasis filter,
// a 400-sample symmetric Hann window centred inside a zero-padded 512-point
// FFT frame, a Slaney mel filterbank with area normalisation, a 2^-24 log
// guard and per-feature mean/std normalisation with N-1 variance. Deviating
// from this (the previous HTK-scale, un-pre-emphasised, left-aligned
// implementation) muffles the high-frequency consonants the encoder relies
// on and makes multilingual audio drift.
const (
	preEmphasis  = 0.97
	logZeroGuard = 1.0 / (1 << 24) // 2^-24, NeMo's log guard
	stdEpsilon   = 1e-5            // added to the std before dividing
)

// MelFilterbank computes mel-scale filterbank features
type MelFilterbank struct {
	nMels      int
	sampleRate int
	nFFT       int
	hopLength  int
	winLength  int
	filterbank [][]float64 // [nMels][nFFT/2+1] continuous Slaney weights
	window     []float64   // length nFFT: Hann(winLength) zero-padded to centre
}

// NewMelFilterbank creates a new mel filterbank extractor
// Using NeMo default parameters for 128 mel features
func NewMelFilterbank(nMels, sampleRate int) *MelFilterbank {
	m := &MelFilterbank{
		nMels:      nMels,
		sampleRate: sampleRate,
		nFFT:       512,
		hopLength:  160, // 10ms at 16kHz
		winLength:  400, // 25ms at 16kHz
	}
	m.filterbank = m.createMelFilterbank()
	m.window = m.createWindow()
	return m
}

// createWindow builds the symmetric Hann window (matching numpy's
// np.hanning(winLength)) zero-padded to nFFT so the non-zero part sits in
// the middle of the FFT frame, exactly like NeMo's centred STFT.
func (m *MelFilterbank) createWindow() []float64 {
	win := make([]float64, m.nFFT)
	pad := (m.nFFT - m.winLength) / 2
	for i := 0; i < m.winLength; i++ {
		win[pad+i] = 0.5 * (1.0 - math.Cos(2.0*math.Pi*float64(i)/float64(m.winLength-1)))
	}
	return win
}

// Slaney mel scale: linear below 1 kHz, logarithmic above. This is the scale
// NeMo/librosa use; the HTK formula the old implementation used places the
// filter centres differently and was one source of the multilingual drift.
const (
	slaneyFreqSpread = 200.0 / 3.0
	slaneyMinLogHz   = 1000.0
	slaneyMinLogMel  = slaneyMinLogHz / slaneyFreqSpread // 15
)

// slaneyLogStep is the per-mel log step above 1 kHz in the Slaney scale.
var slaneyLogStep = math.Log(6.4) / 27.0

func slaneyHzToMel(hz float64) float64 {
	if hz < slaneyMinLogHz {
		return hz / slaneyFreqSpread
	}
	return slaneyMinLogMel + math.Log(hz/slaneyMinLogHz)/slaneyLogStep
}

func slaneyMelToHz(mel float64) float64 {
	if mel < slaneyMinLogMel {
		return slaneyFreqSpread * mel
	}
	return slaneyMinLogHz * math.Exp(slaneyLogStep*(mel-slaneyMinLogMel))
}

// createMelFilterbank builds the Slaney mel filterbank: nMels+2 centre
// frequencies evenly spaced on the mel scale between 0 Hz and Nyquist,
// triangular weights evaluated continuously at each FFT bin frequency (no
// integer-bin quantisation), each filter area-normalised by 2/(f[i+2]-f[i]).
// It matches onnx-asr's fbanks.npz["nemo128"] to ~3e-8.
func (m *MelFilterbank) createMelFilterbank() [][]float64 {
	numBins := m.nFFT/2 + 1
	nyquist := float64(m.sampleRate) / 2.0

	hzPoints := make([]float64, m.nMels+2)
	melMin := slaneyHzToMel(0)
	melMax := slaneyHzToMel(nyquist)
	for i := range hzPoints {
		mel := melMin + float64(i)*(melMax-melMin)/float64(m.nMels+1)
		hzPoints[i] = slaneyMelToHz(mel)
	}

	filterbank := make([][]float64, m.nMels)
	for i := 0; i < m.nMels; i++ {
		filter := make([]float64, numBins)
		lowerSlope := hzPoints[i+1] - hzPoints[i]
		upperSlope := hzPoints[i+2] - hzPoints[i+1]
		for j := 0; j < numBins; j++ {
			freq := float64(j) * float64(m.sampleRate) / float64(m.nFFT)
			rising := (freq - hzPoints[i]) / lowerSlope
			falling := (hzPoints[i+2] - freq) / upperSlope
			if w := math.Min(rising, falling); w > 0 {
				filter[j] = w
			}
		}
		enorm := 2.0 / (hzPoints[i+2] - hzPoints[i])
		for j := range filter {
			filter[j] *= enorm
		}
		filterbank[i] = filter
	}

	return filterbank
}

// FramesPerSecond returns how many mel frames one second of audio yields, set
// by the hop length and sample rate. It ties frame counts to wall-clock time
// so chunk sizes can be configured in seconds.
func (m *MelFilterbank) FramesPerSecond() int {
	return m.sampleRate / m.hopLength
}

// HopLength returns the hop between mel frames in samples. It maps a mel-frame
// index to a sample offset (frame * hopLength), which the VAD boundary oracle
// uses to slice the waveform for an overlap region.
func (m *MelFilterbank) HopLength() int {
	return m.hopLength
}

// Extract computes mel filterbank features from audio samples, matching
// NeMo's AudioToMelSpectrogramPreprocessor (center=True, snip_edges=False):
// the waveform is pre-emphasised, zero-padded by nFFT/2 on both sides, and
// framed with a hop of hopLength, yielding len(samples)/hopLength frames.
func (m *MelFilterbank) Extract(samples []float32) [][]float32 {
	numFrames := len(samples) / m.hopLength
	if numFrames <= 0 {
		if DebugMode {
			slog.Debug("not enough samples for mel frame", "samples", len(samples), "hopLength", m.hopLength)
		}
		return nil
	}

	// 1. Pre-emphasis over the whole waveform: y[0] = x[0];
	//    y[t] = x[t] - 0.97 * x[t-1]. This lifts the high-frequency
	//    consonant formants (s, ß, sch, ch, t, z) that the model needs.
	pre := make([]float64, len(samples))
	var prev float64
	for i, s := range samples {
		cur := float64(s)
		pre[i] = cur - preEmphasis*prev
		prev = cur
	}

	// 2. Frame the pre-emphasised signal with nFFT/2 zero padding on each
	//    side and apply the centred window.
	pad := m.nFFT / 2
	features := make([][]float32, numFrames)
	windowed := make([]float64, m.nFFT)
	numBins := m.nFFT/2 + 1

	for frame := 0; frame < numFrames; frame++ {
		base := frame*m.hopLength - pad
		for i := 0; i < m.nFFT; i++ {
			idx := base + i
			if idx >= 0 && idx < len(pre) {
				windowed[i] = pre[idx] * m.window[i]
			} else {
				windowed[i] = 0
			}
		}

		// FFT
		spectrum := m.fft(windowed)

		// Power spectrum applied through the mel filterbank
		melEnergies := make([]float32, m.nMels)
		for i := 0; i < m.nMels; i++ {
			var energy float64
			fb := m.filterbank[i]
			for j := 0; j < numBins; j++ {
				re := real(spectrum[j])
				im := imag(spectrum[j])
				energy += (re*re + im*im) * fb[j]
			}
			melEnergies[i] = float32(math.Log(energy + logZeroGuard))
		}

		features[frame] = melEnergies
	}

	// 3. Per-feature normalisation over the whole utterance.
	m.normalize(features)

	return features
}

// normalize standardises each mel bin across frames with the sample
// variance (N-1) and adds stdEpsilon to the denominator, matching NeMo's
// normalize="per_feature".
func (m *MelFilterbank) normalize(features [][]float32) {
	if len(features) < 2 {
		return
	}

	n := float64(len(features))
	nFeatures := len(features[0])

	for i := 0; i < nFeatures; i++ {
		var sum float64
		for _, frame := range features {
			sum += float64(frame[i])
		}
		mean := sum / n

		var varSum float64
		for _, frame := range features {
			d := float64(frame[i]) - mean
			varSum += d * d
		}
		denom := math.Sqrt(varSum/(n-1)) + stdEpsilon

		for _, frame := range features {
			frame[i] = float32((float64(frame[i]) - mean) / denom)
		}
	}
}

// fft performs a simple radix-2 FFT
func (m *MelFilterbank) fft(signal []float64) []complex128 {
	n := len(signal)
	if n == 0 {
		return nil
	}

	// Pad to power of 2 if needed
	paddedLen := 1
	for paddedLen < n {
		paddedLen *= 2
	}

	x := make([]complex128, paddedLen)
	for i := 0; i < n; i++ {
		x[i] = complex(signal[i], 0)
	}

	// Bit reversal
	bits := int(math.Log2(float64(paddedLen)))
	for i := 0; i < paddedLen; i++ {
		rev := reverseBits(i, bits)
		if i < rev {
			x[i], x[rev] = x[rev], x[i]
		}
	}

	// Cooley-Tukey FFT
	for size := 2; size <= paddedLen; size *= 2 {
		half := size / 2
		step := -math.Pi / float64(half)
		for i := 0; i < paddedLen; i += size {
			for j := 0; j < half; j++ {
				angle := step * float64(j)
				w := cmplx.Exp(complex(0, angle))
				t := w * x[i+j+half]
				x[i+j+half] = x[i+j] - t
				x[i+j] = x[i+j] + t
			}
		}
	}

	return x
}

func reverseBits(n, bits int) int {
	result := 0
	for i := 0; i < bits; i++ {
		result = (result << 1) | (n & 1)
		n >>= 1
	}
	return result
}
