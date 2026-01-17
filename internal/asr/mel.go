package asr

import (
	"log"
	"math"
	"math/cmplx"
)

// MelFilterbank computes mel-scale filterbank features
type MelFilterbank struct {
	nMels      int
	sampleRate int
	nFFT       int
	hopLength  int
	winLength  int
	filterbank [][]float64
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
	return m
}

func (m *MelFilterbank) hzToMel(hz float64) float64 {
	return 2595.0 * math.Log10(1.0+hz/700.0)
}

func (m *MelFilterbank) melToHz(mel float64) float64 {
	return 700.0 * (math.Pow(10.0, mel/2595.0) - 1.0)
}

func (m *MelFilterbank) createMelFilterbank() [][]float64 {
	numBins := m.nFFT/2 + 1
	melMin := m.hzToMel(0)
	melMax := m.hzToMel(float64(m.sampleRate) / 2)

	// Create mel points
	melPoints := make([]float64, m.nMels+2)
	for i := 0; i <= m.nMels+1; i++ {
		melPoints[i] = melMin + float64(i)*(melMax-melMin)/float64(m.nMels+1)
	}

	// Convert to bin indices
	binPoints := make([]int, m.nMels+2)
	for i, mel := range melPoints {
		hz := m.melToHz(mel)
		binPoints[i] = int(math.Floor(float64(m.nFFT+1) * hz / float64(m.sampleRate)))
	}

	// Create filterbank
	filterbank := make([][]float64, m.nMels)
	for i := 0; i < m.nMels; i++ {
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

// Extract computes mel filterbank features from audio samples
func (m *MelFilterbank) Extract(samples []float32) [][]float32 {
	numFrames := (len(samples)-m.winLength)/m.hopLength + 1
	if numFrames <= 0 {
		if DebugMode {
			log.Printf("[DEBUG] Mel: not enough samples for even one frame (samples=%d, winLength=%d)", len(samples), m.winLength)
		}
		return nil
	}

	features := make([][]float32, numFrames)

	for frame := 0; frame < numFrames; frame++ {
		start := frame * m.hopLength
		end := start + m.winLength
		if end > len(samples) {
			end = len(samples)
		}

		// Extract frame and apply Hann window
		windowed := make([]float64, m.nFFT)
		for i := 0; i < end-start && i < m.winLength; i++ {
			// Hann window
			w := 0.5 * (1.0 - math.Cos(2.0*math.Pi*float64(i)/float64(m.winLength-1)))
			windowed[i] = float64(samples[start+i]) * w
		}

		// FFT
		spectrum := m.fft(windowed)

		// Power spectrum
		numBins := m.nFFT/2 + 1
		power := make([]float64, numBins)
		for i := 0; i < numBins; i++ {
			power[i] = real(spectrum[i])*real(spectrum[i]) + imag(spectrum[i])*imag(spectrum[i])
		}

		// Apply mel filterbank
		melEnergies := make([]float32, m.nMels)
		for i := 0; i < m.nMels; i++ {
			var energy float64
			for j := 0; j < numBins; j++ {
				energy += power[j] * m.filterbank[i][j]
			}
			// Log mel energy
			if energy < 1e-10 {
				energy = 1e-10
			}
			melEnergies[i] = float32(math.Log(energy))
		}

		features[frame] = melEnergies
	}

	// Normalize (optional but helpful)
	m.normalize(features)

	return features
}

func (m *MelFilterbank) normalize(features [][]float32) {
	if len(features) == 0 {
		return
	}

	// Compute mean and std per feature
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
			diff := float64(frame[i]) - means[i]
			sumSq += diff * diff
		}
		stds[i] = math.Sqrt(sumSq / float64(len(features)))
		if stds[i] < 1e-10 {
			stds[i] = 1e-10
		}
	}

	// Apply normalization
	for _, frame := range features {
		for i := 0; i < nFeatures; i++ {
			frame[i] = float32((float64(frame[i]) - means[i]) / stds[i])
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
