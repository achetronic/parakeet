// SPDX-FileCopyrightText: 2026 Alby Hernández <hola@achetronic.com>
// SPDX-License-Identifier: Apache-2.0

//go:build linux && cgo

package memtrim

/*
#include <stddef.h>

// Weak, so a libc without malloc_trim (musl) still links: the symbol resolves to
// NULL there and the check below skips the call.
extern void malloc_trim(size_t) __attribute__((weak));

static void call_malloc_trim(void) {
	if (malloc_trim != NULL) {
		malloc_trim(0);
	}
}
*/
import "C"

// Release returns the free memory of the C heap to the OS. It is a no-op when
// the libc has no malloc_trim.
func Release() {
	C.call_malloc_trim()
}
