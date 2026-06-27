package asr

import (
	"strings"
	"testing"
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
	_, err := ParseProvider("rocm")
	if err == nil {
		t.Fatal("expected error for unsupported provider")
	}
	for _, want := range []string{"cpu", "cuda"} {
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

func TestProviderDefaultsToCPU(t *testing.T) {
	if got := provider(GPUConfig{}); got != ProviderCPU {
		t.Errorf("provider(zero) = %q, want %q", got, ProviderCPU)
	}
	if got := provider(GPUConfig{Provider: ProviderCUDA}); got != ProviderCUDA {
		t.Errorf("provider(cuda) = %q, want %q", got, ProviderCUDA)
	}
}
