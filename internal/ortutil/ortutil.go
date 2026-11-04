// SPDX-FileCopyrightText: 2026 Alby Hernández <hola@achetronic.com>
// SPDX-License-Identifier: Apache-2.0

// Package ortutil centralizes the ONNX Runtime bootstrap shared by the asr and
// diarization packages: locating the shared library and initializing the
// process-wide environment exactly once.
package ortutil

import (
	"fmt"
	"os"

	ort "github.com/yalue/onnxruntime_go"
)

// ResolveLibPath returns the ONNX Runtime shared library path, honoring
// ONNXRUNTIME_LIB first.
func ResolveLibPath() (string, error) {
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

// Init sets the shared library path and initializes the ONNX Runtime
// environment once per process. It returns true when it performed the
// initialization (the caller then owns ort.DestroyEnvironment on shutdown),
// and false when the environment was already initialized elsewhere.
func Init() (bool, error) {
	libPath, err := ResolveLibPath()
	if err != nil {
		return false, err
	}
	ort.SetSharedLibraryPath(libPath)
	if ort.IsInitialized() {
		return false, nil
	}
	if err := ort.InitializeEnvironment(); err != nil {
		return false, fmt.Errorf("initialize ONNX Runtime: %w", err)
	}
	return true, nil
}
