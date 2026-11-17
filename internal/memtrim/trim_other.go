// SPDX-FileCopyrightText: 2026 Alby Hernández <hola@achetronic.com>
// SPDX-License-Identifier: Apache-2.0

//go:build !linux || !cgo

package memtrim

// Release is a no-op: this platform has no malloc_trim to call.
func Release() {}
