// SPDX-FileCopyrightText: 2026 Alby Hernández <hola@achetronic.com>
// SPDX-License-Identifier: Apache-2.0

package ortutil

import (
	"testing"

	ort "github.com/yalue/onnxruntime_go"
)

func TestInitIdempotent(t *testing.T) {
	if _, err := ResolveLibPath(); err != nil {
		t.Skip("onnxruntime library not installed")
	}

	first, err := Init()
	if err != nil {
		t.Fatalf("first Init: %v", err)
	}
	if !first {
		t.Fatal("first Init should initialize the environment")
	}

	second, err := Init()
	if err != nil {
		t.Fatalf("second Init: %v", err)
	}
	if second {
		t.Fatal("second Init should report the environment already initialized")
	}

	if err := ort.DestroyEnvironment(); err != nil {
		t.Fatalf("DestroyEnvironment: %v", err)
	}
}
