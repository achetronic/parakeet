// SPDX-FileCopyrightText: 2026 Alby Hernández <hola@achetronic.com>
// SPDX-License-Identifier: Apache-2.0

package diarization

import (
	"fmt"
	"os"
	"path/filepath"

	ort "github.com/yalue/onnxruntime_go"
)

const (
	inputFeaturesName = "input_features"
	cachedEmbedsName  = "cached_embeds"
	attentionMaskName = "attention_mask"
	logitsName        = "logits"
	chunkEmbedsName   = "chunk_embeds"
	silenceEmbedsName = "silence_embeds"
)

// resolveLibPath returns the ONNX Runtime shared library path, honoring
// ONNXRUNTIME_LIB first. Mirrors the lookup order used by the asr transcriber.
func resolveLibPath() (string, error) {
	if p := os.Getenv("ONNXRUNTIME_LIB"); p != "" {
		return p, nil
	}
	for _, p := range []string{
		"/usr/lib/libonnxruntime.so",
		"/usr/lib/x86_64-linux-gnu/libonnxruntime.so",
		"/usr/local/lib/libonnxruntime.so",
		"/opt/onnxruntime/lib/libonnxruntime.so",
		"./libonnxruntime.so",
		"libonnxruntime.so.1.25.1",
	} {
		if _, err := os.Stat(p); err == nil {
			return p, nil
		}
	}
	return "", fmt.Errorf("ONNX Runtime library not found. Set ONNXRUNTIME_LIB env var or install libonnxruntime")
}

// resolveModelPath returns the int8 model when present, else the fp32 one.
func resolveModelPath(dir string) (string, error) {
	for _, name := range []string{"model_quantized.onnx", "model.onnx"} {
		p := filepath.Join(dir, name)
		if _, err := os.Stat(p); err == nil {
			return p, nil
		}
	}
	return "", fmt.Errorf("diarization model not found in %s", dir)
}

// loadSession creates the dynamic ONNX session for the diarization encoder.
func loadSession(modelPath string) (*ort.DynamicAdvancedSession, error) {
	sess, err := ort.NewDynamicAdvancedSession(
		modelPath,
		[]string{inputFeaturesName, cachedEmbedsName, attentionMaskName},
		[]string{logitsName, chunkEmbedsName, silenceEmbedsName},
		nil,
	)
	if err != nil {
		return nil, fmt.Errorf("create diarization session: %w", err)
	}
	return sess, nil
}
