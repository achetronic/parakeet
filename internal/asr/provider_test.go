// SPDX-FileCopyrightText: 2026 Alby Hernández <hola@achetronic.com>
// SPDX-License-Identifier: Apache-2.0

package asr

import (
	"reflect"
	"strings"
	"testing"

	ort "github.com/yalue/onnxruntime_go"
)

func TestParseProvider(t *testing.T) {
	cases := []struct {
		name    string
		in      string
		want    Provider
		wantErr bool
	}{
		{"empty defaults to cpu", "", ProviderCPU, false},
		{"explicit cpu", "cpu", ProviderCPU, false},
		{"cuda", "cuda", ProviderCUDA, false},
		{"rocm", "rocm", ProviderROCm, false},
		{"uppercase normalized", "CUDA", ProviderCUDA, false},
		{"surrounding whitespace", "  cuda  ", ProviderCUDA, false},
		{"unknown rejected", "tensorrt", "", true},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got, err := ParseProvider(tc.in)
			if tc.wantErr {
				if err == nil {
					t.Fatalf("ParseProvider(%q) = %q, want error", tc.in, got)
				}
				return
			}
			if err != nil {
				t.Fatalf("ParseProvider(%q) unexpected error: %v", tc.in, err)
			}
			if got != tc.want {
				t.Errorf("ParseProvider(%q) = %q, want %q", tc.in, got, tc.want)
			}
		})
	}
}

func TestParseProviderErrorNamesSupported(t *testing.T) {
	_, err := ParseProvider("openvino")
	if err == nil {
		t.Fatal("expected error for unsupported provider")
	}
	for _, want := range []string{"cpu", "cuda", "rocm"} {
		if !strings.Contains(err.Error(), want) {
			t.Errorf("error %q should name supported provider %q", err.Error(), want)
		}
	}
}

// buildSessionOptions must not touch the ONNX Runtime for the CPU provider:
// it returns (nil, nil) so sessions are created with default CPU behavior,
// identical to the pre-GPU path. The CUDA branch requires a loaded runtime and
// a GPU, so it is exercised manually, not in CI (see spec acceptance criteria).
func TestBuildSessionOptionsCPU(t *testing.T) {
	for _, p := range []Provider{ProviderCPU, Provider("")} {
		opts, err := buildSessionOptions(GPUConfig{Provider: p})
		if err != nil {
			t.Fatalf("buildSessionOptions(%q) error: %v", p, err)
		}
		if opts != nil {
			t.Errorf("buildSessionOptions(%q) = %v, want nil", p, opts)
		}
	}
}

// The ROCm shim (rocm_ep.c) extracts the raw OrtSessionOptions pointer from
// the binding's SessionOptions value, which relies on that struct being a
// single-pointer struct. If a binding upgrade changes the layout, this test
// fails loudly instead of the shim silently corrupting memory.
func TestSessionOptionsShimLayout(t *testing.T) {
	tp := reflect.TypeOf(ort.SessionOptions{})
	if tp.NumField() != 1 || tp.Field(0).Type.Kind() != reflect.Ptr {
		t.Fatalf("ROCm shim assumes ort.SessionOptions is a single-pointer struct, got %d field(s): %v",
			tp.NumField(), tp)
	}
}

func TestProviderDefaultsToCPU(t *testing.T) {
	if got := provider(GPUConfig{}); got != ProviderCPU {
		t.Errorf("provider(zero) = %q, want %q", got, ProviderCPU)
	}
	if got := provider(GPUConfig{Provider: ProviderCUDA}); got != ProviderCUDA {
		t.Errorf("provider(cuda) = %q, want %q", got, ProviderCUDA)
	}
}
