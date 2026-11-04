// SPDX-FileCopyrightText: 2026 Alby Hernández <hola@achetronic.com>
// SPDX-License-Identifier: Apache-2.0

// Package audio decodes request bytes into mono 16 kHz float32 samples: WAV
// in-process, anything else through an optional ffmpeg converter.
package audio

import (
	"encoding/binary"
	"errors"
	"fmt"
	"log/slog"
	"math"
)

// Debug gates verbose logging. Set by the server based on log level.
var Debug bool

// ErrUnsupportedAudio is returned when the input is neither a parsable WAV
// nor convertible via ffmpeg. Callers use errors.Is to map it to HTTP 400.
var ErrUnsupportedAudio = errors.New("unsupported audio")

// Load decodes raw request bytes into mono 16 kHz float32 samples. WAV is
// parsed in-process; anything else is delegated to the optional ffmpeg
// converter, or rejected with ErrUnsupportedAudio when converter is nil.
func Load(data []byte, converter *Converter) ([]float32, error) {
	if isWAV(data) {
		return parseWAV(data)
	}
	if converter == nil {
		return nil, fmt.Errorf("input is not WAV and ffmpeg conversion is disabled: %w", ErrUnsupportedAudio)
	}
	if Debug {
		slog.Debug("converting audio via ffmpeg", "bytes", len(data))
	}
	wavData, err := converter.Convert(data)
	if err != nil {
		return nil, err
	}
	return parseWAV(wavData)
}

// isWAV returns true when data starts with a RIFF/WAVE header.
func isWAV(data []byte) bool {
	if len(data) < 12 {
		return false
	}
	return string(data[0:4]) == "RIFF" && string(data[8:12]) == "WAVE"
}

// parseWAV parses a WAV file and returns float32 samples normalized to [-1, 1].
func parseWAV(data []byte) ([]float32, error) {
	if len(data) < 44 {
		return nil, fmt.Errorf("WAV file too small")
	}
	if string(data[0:4]) != "RIFF" {
		return nil, fmt.Errorf("not a RIFF file")
	}
	if string(data[8:12]) != "WAVE" {
		return nil, fmt.Errorf("not a WAVE file")
	}

	offset := 12
	var audioFormat, numChannels uint16
	var sampleRate uint32
	var bitsPerSample uint16

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
			bitsPerSample = binary.LittleEndian.Uint16(data[offset+22 : offset+24])
		} else if chunkID == "data" {
			dataStart := offset + 8
			dataEnd := dataStart + int(chunkSize)
			if dataEnd > len(data) {
				dataEnd = len(data)
			}
			audioData := data[dataStart:dataEnd]

			if Debug {
				slog.Debug("WAV parsed",
					"format", audioFormat,
					"channels", numChannels,
					"sampleRate", sampleRate,
					"bitsPerSample", bitsPerSample,
					"dataSize", len(audioData),
				)
			}

			samples, err := convertToFloat32(audioData, audioFormat, numChannels, bitsPerSample)
			if err != nil {
				return nil, err
			}
			if sampleRate != 16000 {
				if Debug {
					slog.Debug("resampling",
						"from", sampleRate,
						"to", 16000,
						"samplesIn", len(samples),
						"samplesOut", int(float64(len(samples))*16000.0/float64(sampleRate)),
					)
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
				val = float64(data[offset])/128.0 - 1.0
			case 16:
				sample := int16(binary.LittleEndian.Uint16(data[offset : offset+2]))
				val = float64(sample) / 32768.0
			case 24:
				b := data[offset : offset+3]
				sample := int32(b[0]) | int32(b[1])<<8 | int32(b[2])<<16
				if sample&0x800000 != 0 {
					sample |= ^0xffffff // Sign extend
				}
				val = float64(sample) / 8388608.0
			case 32:
				if audioFormat == 3 {
					bits := binary.LittleEndian.Uint32(data[offset : offset+4])
					val = float64(math.Float32frombits(bits))
				} else {
					sample := int32(binary.LittleEndian.Uint32(data[offset : offset+4]))
					val = float64(sample) / 2147483648.0
				}
			default:
				return nil, fmt.Errorf("unsupported bits per sample: %d", bitsPerSample)
			}
			sum += val
		}
		samples[i] = float32(sum / float64(numChannels))
	}
	return samples, nil
}

// resample uses linear interpolation for simple resampling.
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
