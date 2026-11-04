// SPDX-FileCopyrightText: 2026 Alby Hernández <hola@achetronic.com>
// SPDX-License-Identifier: Apache-2.0

package diarization

import (
	"math"
	"math/cmplx"
)

// featureExtractor computes log-mel filterbank features with pre-emphasis, as
// the Nemotron-3-Diarization model expects them. It is deliberately kept in
// this package rather than reusing asr's mel filterbank: the two models use
// different front-ends (this one applies pre-emphasis 0.97).
type featureExtractor struct {
	cfg        Config
	filterbank [][]float64
	hannWindow []float64
	numFFTBins int
}

func newFeatureExtractor(cfg Config) *featureExtractor {
	e := &featureExtractor{cfg: cfg}
	e.filterbank = e.buildFilterbank()
	e.hannWindow = e.buildHannWindow()
	e.numFFTBins = cfg.NFFT/2 + 1
	return e
}

func (e *featureExtractor) buildHannWindow() []float64 {
	win := make([]float64, e.cfg.WinLength)
	for i := range win {
		win[i] = 0.5 * (1.0 - math.Cos(2.0*math.Pi*float64(i)/float64(e.cfg.WinLength-1)))
	}
	return win
}

func hzToMel(hz float64) float64  { return 2595.0 * math.Log10(1.0+hz/700.0) }
func melToHz(mel float64) float64 { return 700.0 * (math.Pow(10.0, mel/2595.0) - 1.0) }

func (e *featureExtractor) buildFilterbank() [][]float64 {
	numBins := e.cfg.NFFT/2 + 1
	melMin := hzToMel(0)
	melMax := hzToMel(float64(e.cfg.SampleRate) / 2)

	melPoints := make([]float64, e.cfg.FeatureSize+2)
	for i := range melPoints {
		melPoints[i] = melMin + float64(i)*(melMax-melMin)/float64(e.cfg.FeatureSize+1)
	}
	binPoints := make([]int, e.cfg.FeatureSize+2)
	for i, mel := range melPoints {
		binPoints[i] = int(math.Floor(float64(e.cfg.NFFT+1) * melToHz(mel) / float64(e.cfg.SampleRate)))
	}

	filterbank := make([][]float64, e.cfg.FeatureSize)
	for i := 0; i < e.cfg.FeatureSize; i++ {
		filter := make([]float64, numBins)
		for j := binPoints[i]; j < binPoints[i+1] && j < numBins; j++ {
			filter[j] = float64(j-binPoints[i]) / float64(binPoints[i+1]-binPoints[i])
		}
		for j := binPoints[i+1]; j < binPoints[i+2] && j < numBins; j++ {
			filter[j] = float64(binPoints[i+2]-j) / float64(binPoints[i+2]-binPoints[i+1])
		}
		filterbank[i] = filter
	}
	return filterbank
}

// preemphasis applies y[n] = x[n] - k*x[n-1] in place over a copy.
func (e *featureExtractor) preemphasis(samples []float32) []float32 {
	out := make([]float32, len(samples))
	k := float32(e.cfg.Preemphasis)
	for i := 1; i < len(samples); i++ {
		out[i] = samples[i] - k*samples[i-1]
	}
	if len(samples) > 0 {
		out[0] = samples[0]
	}
	return out
}

// extract returns log-mel features [numFrames][featureSize] with per-feature
// mean/std normalization over the whole utterance.
func (e *featureExtractor) extract(samples []float32) [][]float32 {
	samples = e.preemphasis(samples)

	numFrames := (len(samples)-e.cfg.WinLength)/e.cfg.HopLength + 1
	if numFrames <= 0 {
		return nil
	}

	features := make([][]float32, numFrames)
	for frame := 0; frame < numFrames; frame++ {
		start := frame * e.cfg.HopLength
		windowed := make([]float64, e.cfg.NFFT)
		for i := 0; i < e.cfg.WinLength && start+i < len(samples); i++ {
			windowed[i] = float64(samples[start+i]) * e.hannWindow[i]
		}

		spectrum := fft(windowed)
		melEnergies := make([]float32, e.cfg.FeatureSize)
		for i := 0; i < e.cfg.FeatureSize; i++ {
			var energy float64
			for j := 0; j < e.numFFTBins; j++ {
				energy += (real(spectrum[j])*real(spectrum[j]) + imag(spectrum[j])*imag(spectrum[j])) * e.filterbank[i][j]
			}
			if energy < 1e-10 {
				energy = 1e-10
			}
			melEnergies[i] = float32(math.Log(energy))
		}
		features[frame] = melEnergies
	}

	normalize(features)
	return features
}

func normalize(features [][]float32) {
	if len(features) == 0 {
		return
	}
	nFeatures := len(features[0])
	means := make([]float64, nFeatures)
	stds := make([]float64, nFeatures)
	for i := 0; i < nFeatures; i++ {
		var sum, sumSq float64
		for _, frame := range features {
			sum += float64(frame[i])
		}
		means[i] = sum / float64(len(features))
		for _, frame := range features {
			d := float64(frame[i]) - means[i]
			sumSq += d * d
		}
		stds[i] = math.Sqrt(sumSq / float64(len(features)))
		if stds[i] < 1e-10 {
			stds[i] = 1e-10
		}
	}
	for _, frame := range features {
		for i := 0; i < nFeatures; i++ {
			frame[i] = float32((float64(frame[i]) - means[i]) / stds[i])
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
