// SPDX-FileCopyrightText: 2026 Alby Hernández <hola@achetronic.com>
// SPDX-License-Identifier: Apache-2.0

// Package mel computes log-mel filterbank features matching NeMo's
// AudioToMelSpectrogramPreprocessor: a pre-emphasis filter, a symmetric Hann
// window centred inside a zero-padded FFT frame, a Slaney mel filterbank with
// area normalisation, a 2^-24 log guard and per-feature mean/std normalisation
// with N-1 variance. It is the front-end shared by the asr and diarization
// models.
package mel

import (
	"math"
	"math/cmplx"
)

// NeMo preprocessor constants.
const (
	logZeroGuard = 1.0 / (1 << 24) // 2^-24, NeMo's log guard
	stdEpsilon   = 1e-5            // added to the std before dividing
)

// Filterbank computes log-mel filterbank features.
type Filterbank struct {
	nMels       int
	sampleRate  int
	nFFT        int
	hopLength   int
	winLength   int
	preEmphasis float64
	normalize   bool
	filterbank  [][]float64 // [nMels][nFFT/2+1] continuous Slaney weights
	window      []float64   // length nFFT: Hann(winLength) zero-padded to centre
}

// New creates a mel filterbank extractor. preEmphasis is the pre-emphasis
// coefficient (0.97 for NeMo/Nemotron; 0 disables it). normalize enables the
// per-feature mean/std (N-1) normalization: NeMo ASR wants it, but the
// Nemotron-3 diarization frontend never normalizes the mel features.
func New(nMels, sampleRate, nFFT, hopLength, winLength int, preEmphasis float64, normalize bool) *Filterbank {
	f := &Filterbank{
		nMels:       nMels,
		sampleRate:  sampleRate,
		nFFT:        nFFT,
		hopLength:   hopLength,
		winLength:   winLength,
		preEmphasis: preEmphasis,
		normalize:   normalize,
	}
	f.filterbank = f.createMelFilterbank()
	f.window = f.createWindow()
	return f
}

// createWindow builds the symmetric Hann window (matching numpy's
// np.hanning(winLength)) zero-padded to nFFT so the non-zero part sits in the
// middle of the FFT frame, like NeMo's centred STFT.
func (f *Filterbank) createWindow() []float64 {
	win := make([]float64, f.nFFT)
	pad := (f.nFFT - f.winLength) / 2
	for i := 0; i < f.winLength; i++ {
		win[pad+i] = 0.5 * (1.0 - math.Cos(2.0*math.Pi*float64(i)/float64(f.winLength-1)))
	}
	return win
}

// Slaney mel scale: linear below 1 kHz, logarithmic above.
const (
	slaneyFreqSpread = 200.0 / 3.0
	slaneyMinLogHz   = 1000.0
	slaneyMinLogMel  = slaneyMinLogHz / slaneyFreqSpread // 15
)

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
// frequencies on the mel scale, triangular weights evaluated continuously at
// each FFT bin frequency, each filter area-normalised by 2/(f[i+2]-f[i]).
func (f *Filterbank) createMelFilterbank() [][]float64 {
	numBins := f.nFFT/2 + 1
	nyquist := float64(f.sampleRate) / 2.0

	hzPoints := make([]float64, f.nMels+2)
	melMin := slaneyHzToMel(0)
	melMax := slaneyHzToMel(nyquist)
	for i := range hzPoints {
		melPoints := melMin + float64(i)*(melMax-melMin)/float64(f.nMels+1)
		hzPoints[i] = slaneyMelToHz(melPoints)
	}

	filterbank := make([][]float64, f.nMels)
	for i := 0; i < f.nMels; i++ {
		filter := make([]float64, numBins)
		lowerSlope := hzPoints[i+1] - hzPoints[i]
		upperSlope := hzPoints[i+2] - hzPoints[i+1]
		for j := 0; j < numBins; j++ {
			freq := float64(j) * float64(f.sampleRate) / float64(f.nFFT)
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

// FramesPerSecond returns how many mel frames one second of audio yields.
func (f *Filterbank) FramesPerSecond() int {
	return f.sampleRate / f.hopLength
}

// HopLength returns the hop between mel frames in samples.
func (f *Filterbank) HopLength() int {
	return f.hopLength
}

// Extract computes log-mel filterbank features from audio samples, matching
// NeMo's AudioToMelSpectrogramPreprocessor (center=True, snip_edges=False):
// the waveform is pre-emphasised, zero-padded by nFFT/2 on both sides, and
// framed with a hop of hopLength, yielding len(samples)/hopLength frames.
func (f *Filterbank) Extract(samples []float32) [][]float32 {
	numFrames := len(samples) / f.hopLength
	if numFrames <= 0 {
		return nil
	}

	// 1. Pre-emphasis: y[0] = x[0]; y[t] = x[t] - k*x[t-1].
	pre := make([]float64, len(samples))
	if f.preEmphasis > 0 {
		var prev float64
		for i, s := range samples {
			cur := float64(s)
			pre[i] = cur - f.preEmphasis*prev
			prev = cur
		}
	} else {
		for i, s := range samples {
			pre[i] = float64(s)
		}
	}

	// 2. Frame with nFFT/2 zero padding on each side, centred window.
	pad := f.nFFT / 2
	features := make([][]float32, numFrames)
	windowed := make([]float64, f.nFFT)
	numBins := f.nFFT/2 + 1

	for frame := 0; frame < numFrames; frame++ {
		base := frame*f.hopLength - pad
		for i := 0; i < f.nFFT; i++ {
			idx := base + i
			if idx >= 0 && idx < len(pre) {
				windowed[i] = pre[idx] * f.window[i]
			} else {
				windowed[i] = 0
			}
		}

		spectrum := fft(windowed)

		melEnergies := make([]float32, f.nMels)
		for i := 0; i < f.nMels; i++ {
			var energy float64
			fb := f.filterbank[i]
			for j := 0; j < numBins; j++ {
				re := real(spectrum[j])
				im := imag(spectrum[j])
				energy += (re*re + im*im) * fb[j]
			}
			melEnergies[i] = float32(math.Log(energy + logZeroGuard))
		}
		features[frame] = melEnergies
	}

	// 3. Per-feature normalisation over the whole utterance (NeMo ASR only).
	if f.normalize {
		normalize(features)
	}
	return features
}

// normalize standardises each mel bin across frames with the sample variance
// (N-1) and adds stdEpsilon to the denominator, matching NeMo's
// normalize="per_feature".
func normalize(features [][]float32) {
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

// fft is a radix-2 Cooley-Tukey FFT over a zero-padded input.
func fft(signal []float64) []complex128 {
	n := len(signal)
	if n == 0 {
		return nil
	}
	padded := 1
	for padded < n {
		padded *= 2
	}
	x := make([]complex128, padded)
	for i := 0; i < n; i++ {
		x[i] = complex(signal[i], 0)
	}
	bits := int(math.Log2(float64(padded)))
	for i := 0; i < padded; i++ {
		rev := reverseBits(i, bits)
		if i < rev {
			x[i], x[rev] = x[rev], x[i]
		}
	}
	for size := 2; size <= padded; size *= 2 {
		half := size / 2
		step := -math.Pi / float64(half)
		for i := 0; i < padded; i += size {
			for j := 0; j < half; j++ {
				w := cmplx.Exp(complex(0, step*float64(j)))
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
