// SPDX-FileCopyrightText: 2026 Alby Hernández <hola@achetronic.com>
// SPDX-License-Identifier: Apache-2.0

// Package mel computes log-mel filterbank features: windowing, FFT, power
// spectrum, mel filterbank, log, and per-feature mean/std normalization. It is
// the front-end shared by the asr and diarization models; callers with extra
// preprocessing (e.g. pre-emphasis) apply it before calling Extract.
package mel

import (
	"math"
	"math/cmplx"
)

// Filterbank computes log-mel filterbank features.
type Filterbank struct {
	nMels      int
	sampleRate int
	nFFT       int
	hopLength  int
	winLength  int
	filterbank [][]float64
	hannWindow []float64
}

// New creates a mel filterbank extractor for the given front-end parameters.
func New(nMels, sampleRate, nFFT, hopLength, winLength int) *Filterbank {
	f := &Filterbank{
		nMels:      nMels,
		sampleRate: sampleRate,
		nFFT:       nFFT,
		hopLength:  hopLength,
		winLength:  winLength,
	}
	f.filterbank = f.createMelFilterbank()
	f.hannWindow = f.createHannWindow()
	return f
}

func (f *Filterbank) createHannWindow() []float64 {
	win := make([]float64, f.winLength)
	for i := range win {
		win[i] = 0.5 * (1.0 - math.Cos(2.0*math.Pi*float64(i)/float64(f.winLength-1)))
	}
	return win
}

func hzToMel(hz float64) float64  { return 2595.0 * math.Log10(1.0+hz/700.0) }
func melToHz(mel float64) float64 { return 700.0 * (math.Pow(10.0, mel/2595.0) - 1.0) }

func (f *Filterbank) createMelFilterbank() [][]float64 {
	numBins := f.nFFT/2 + 1
	melMin := hzToMel(0)
	melMax := hzToMel(float64(f.sampleRate) / 2)

	melPoints := make([]float64, f.nMels+2)
	for i := range melPoints {
		melPoints[i] = melMin + float64(i)*(melMax-melMin)/float64(f.nMels+1)
	}
	binPoints := make([]int, f.nMels+2)
	for i, m := range melPoints {
		binPoints[i] = int(math.Floor(float64(f.nFFT+1) * melToHz(m) / float64(f.sampleRate)))
	}

	filterbank := make([][]float64, f.nMels)
	for i := 0; i < f.nMels; i++ {
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

// FramesPerSecond returns how many mel frames one second of audio yields.
func (f *Filterbank) FramesPerSecond() int {
	return f.sampleRate / f.hopLength
}

// HopLength returns the hop between mel frames in samples.
func (f *Filterbank) HopLength() int {
	return f.hopLength
}

// Extract computes log-mel filterbank features from audio samples.
func (f *Filterbank) Extract(samples []float32) [][]float32 {
	numFrames := (len(samples)-f.winLength)/f.hopLength + 1
	if numFrames <= 0 {
		return nil
	}

	features := make([][]float32, numFrames)
	for frame := 0; frame < numFrames; frame++ {
		start := frame * f.hopLength
		windowed := make([]float64, f.nFFT)
		for i := 0; i < f.winLength && start+i < len(samples); i++ {
			windowed[i] = float64(samples[start+i]) * f.hannWindow[i]
		}

		spectrum := fft(windowed)
		numBins := f.nFFT/2 + 1
		melEnergies := make([]float32, f.nMels)
		for i := 0; i < f.nMels; i++ {
			var energy float64
			for j := 0; j < numBins; j++ {
				energy += (real(spectrum[j])*real(spectrum[j]) + imag(spectrum[j])*imag(spectrum[j])) * f.filterbank[i][j]
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
