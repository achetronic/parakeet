package asr

import (
	"encoding/binary"
	"fmt"
	"log"
	"math"
)

// parseWAV parses a WAV file and returns float32 samples normalized to [-1, 1]
func parseWAV(data []byte) ([]float32, error) {
	if len(data) < 44 {
		return nil, fmt.Errorf("WAV file too small")
	}

	// Check RIFF header
	if string(data[0:4]) != "RIFF" {
		return nil, fmt.Errorf("not a RIFF file")
	}
	if string(data[8:12]) != "WAVE" {
		return nil, fmt.Errorf("not a WAVE file")
	}

	// Find fmt chunk
	offset := 12
	var audioFormat, numChannels uint16
	var sampleRate, byteRate uint32
	var blockAlign, bitsPerSample uint16
	
	for offset < len(data)-8 {
		chunkID := string(data[offset : offset+4])
		chunkSize := binary.LittleEndian.Uint32(data[offset+4 : offset+8])
		
		if chunkID == "fmt " {
			if chunkSize < 16 {
				return nil, fmt.Errorf("fmt chunk too small")
			}
			audioFormat = binary.LittleEndian.Uint16(data[offset+8 : offset+10])
			numChannels = binary.LittleEndian.Uint16(data[offset+10 : offset+12])
			sampleRate = binary.LittleEndian.Uint32(data[offset+12 : offset+16])
			byteRate = binary.LittleEndian.Uint32(data[offset+16 : offset+20])
			blockAlign = binary.LittleEndian.Uint16(data[offset+20 : offset+22])
			bitsPerSample = binary.LittleEndian.Uint16(data[offset+22 : offset+24])
			_ = byteRate   // unused
			_ = blockAlign // unused
		} else if chunkID == "data" {
			dataStart := offset + 8
			dataEnd := dataStart + int(chunkSize)
			if dataEnd > len(data) {
				dataEnd = len(data)
			}
			audioData := data[dataStart:dataEnd]

			if DebugMode {
				log.Printf("[DEBUG] WAV: format=%d, channels=%d, sampleRate=%d, bitsPerSample=%d, dataSize=%d",
					audioFormat, numChannels, sampleRate, bitsPerSample, len(audioData))
			}

			// Convert to float32
			samples, err := convertToFloat32(audioData, audioFormat, numChannels, bitsPerSample)
			if err != nil {
				return nil, err
			}

			// Resample to 16kHz if needed
			if sampleRate != 16000 {
				if DebugMode {
					log.Printf("[DEBUG] Resampling from %d Hz to 16000 Hz (%d -> %d samples)",
						sampleRate, len(samples), int(float64(len(samples))*16000.0/float64(sampleRate)))
				}
				samples = resample(samples, int(sampleRate), 16000)
			}

			return samples, nil
		}
		
		offset += 8 + int(chunkSize)
		if chunkSize%2 != 0 {
			offset++ // Padding byte
		}
	}
	
	return nil, fmt.Errorf("no data chunk found")
}

func convertToFloat32(data []byte, audioFormat, numChannels, bitsPerSample uint16) ([]float32, error) {
	if audioFormat != 1 && audioFormat != 3 {
		return nil, fmt.Errorf("unsupported audio format: %d (only PCM supported)", audioFormat)
	}

	bytesPerSample := int(bitsPerSample / 8)
	numSamples := len(data) / (bytesPerSample * int(numChannels))
	samples := make([]float32, numSamples)

	for i := 0; i < numSamples; i++ {
		var sum float64
		for ch := 0; ch < int(numChannels); ch++ {
			offset := (i*int(numChannels) + ch) * bytesPerSample
			if offset+bytesPerSample > len(data) {
				break
			}

			var val float64
			switch bitsPerSample {
			case 8:
				// Unsigned 8-bit
				val = float64(data[offset])/128.0 - 1.0
			case 16:
				// Signed 16-bit little endian
				sample := int16(binary.LittleEndian.Uint16(data[offset : offset+2]))
				val = float64(sample) / 32768.0
			case 24:
				// Signed 24-bit little endian
				b := data[offset : offset+3]
				sample := int32(b[0]) | int32(b[1])<<8 | int32(b[2])<<16
				if sample&0x800000 != 0 {
					sample |= ^0xffffff // Sign extend
				}
				val = float64(sample) / 8388608.0
			case 32:
				if audioFormat == 3 {
					// Float 32-bit
					bits := binary.LittleEndian.Uint32(data[offset : offset+4])
					val = float64(math.Float32frombits(bits))
				} else {
					// Signed 32-bit
					sample := int32(binary.LittleEndian.Uint32(data[offset : offset+4]))
					val = float64(sample) / 2147483648.0
				}
			default:
				return nil, fmt.Errorf("unsupported bits per sample: %d", bitsPerSample)
			}
			sum += val
		}
		// Average channels (convert stereo to mono)
		samples[i] = float32(sum / float64(numChannels))
	}

	return samples, nil
}

// resample uses linear interpolation for simple resampling
func resample(samples []float32, srcRate, dstRate int) []float32 {
	if srcRate == dstRate {
		return samples
	}

	ratio := float64(srcRate) / float64(dstRate)
	newLen := int(float64(len(samples)) / ratio)
	result := make([]float32, newLen)

	for i := 0; i < newLen; i++ {
		srcIdx := float64(i) * ratio
		lo := int(srcIdx)
		hi := lo + 1
		if hi >= len(samples) {
			hi = len(samples) - 1
		}
		frac := float32(srcIdx - float64(lo))
		result[i] = samples[lo]*(1-frac) + samples[hi]*frac
	}

	return result
}
