// SPDX-FileCopyrightText: 2026 Alby Hernández <hola@achetronic.com>
// SPDX-License-Identifier: Apache-2.0

package server

import "testing"

func TestParseGPUDevices(t *testing.T) {
	cases := []struct {
		name    string
		in      string
		want    []int
		wantErr bool
	}{
		{"empty", "", nil, false},
		{"single", "0", []int{0}, false},
		{"list", "0,1,2", []int{0, 1, 2}, false},
		{"whitespace tolerated", " 0 , 1 ", []int{0, 1}, false},
		{"blank entry rejected", "0,,1", nil, true},
		{"non-numeric rejected", "a", nil, true},
		{"negative rejected", "-1", nil, true},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got, err := ParseGPUDevices(tc.in)
			if tc.wantErr {
				if err == nil {
					t.Fatalf("ParseGPUDevices(%q) = %v, want error", tc.in, got)
				}
				return
			}
			if err != nil {
				t.Fatalf("ParseGPUDevices(%q) error: %v", tc.in, err)
			}
			if len(got) != len(tc.want) {
				t.Fatalf("ParseGPUDevices(%q) = %v, want %v", tc.in, got, tc.want)
			}
			for i := range got {
				if got[i] != tc.want[i] {
					t.Fatalf("ParseGPUDevices(%q) = %v, want %v", tc.in, got, tc.want)
				}
			}
		})
	}
}
