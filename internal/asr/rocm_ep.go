// SPDX-FileCopyrightText: 2026 Alby Hernández <hola@achetronic.com>
// SPDX-License-Identifier: Apache-2.0

//go:build !windows

package asr

/*
#cgo LDFLAGS: -ldl

#include <stdlib.h>

int AppendROCmProvider(const char *lib_path, void *session_options,
                       int device_id, char *errbuf, size_t errbuf_len);
*/
import "C"

import (
	"fmt"
	"unsafe"

	ort "github.com/yalue/onnxruntime_go"
)

// appendROCmExecutionProvider enables the ROCm execution provider on opts
// for the given GPU device.
//
// The Go binding has no ROCm wrapper and ORT's generic name-based
// AppendExecutionProvider rejects "ROCM" (its supported list covers
// DML/QNN/OpenVINO/... only), so the provider is enabled through a small
// cgo shim that calls the dedicated
// OrtSessionOptionsAppendExecutionProvider_ROCM entry point exported by
// ROCm-enabled ONNX Runtime builds (see rocm_ep.c).
func appendROCmExecutionProvider(opts *ort.SessionOptions, deviceID int) error {
	libPath := resolveORTLibPath()
	if libPath == "" {
		return fmt.Errorf("ONNX Runtime library not found")
	}
	// ort.SessionOptions is a single-pointer struct wrapping the underlying
	// *OrtSessionOptions; hand that pointer to the C shim. If a future
	// binding upgrade changes that layout, TestSessionOptionsShimLayout
	// fails loudly.
	soPtr := *(*unsafe.Pointer)(unsafe.Pointer(opts))
	var errbuf [512]C.char
	cLib := C.CString(libPath)
	defer C.free(unsafe.Pointer(cLib))
	if C.AppendROCmProvider(cLib, soPtr, C.int(deviceID), &errbuf[0],
		C.size_t(len(errbuf))) != 0 {
		return fmt.Errorf("%s", C.GoString(&errbuf[0]))
	}
	return nil
}
