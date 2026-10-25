// SPDX-FileCopyrightText: 2026 Alby Hernández <hola@achetronic.com>
// SPDX-License-Identifier: Apache-2.0

//go:build windows

package asr

import (
	"fmt"

	ort "github.com/yalue/onnxruntime_go"
)

// appendROCmExecutionProvider is unavailable on Windows: AMD ships
// ROCm-enabled ONNX Runtime builds for Linux only, so the dlopen-based shim
// used on other platforms has no Windows counterpart.
func appendROCmExecutionProvider(opts *ort.SessionOptions, deviceID int) error {
	return fmt.Errorf("ROCm execution provider is not supported on Windows")
}
