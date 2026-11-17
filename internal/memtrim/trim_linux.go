// SPDX-FileCopyrightText: 2026 Alby Hernández <hola@achetronic.com>
// SPDX-License-Identifier: Apache-2.0

//go:build linux && cgo

package memtrim

/*
#include <dlfcn.h>
#include <stddef.h>

// Look the symbol up at run time instead of linking it: musl has no
// malloc_trim, and a direct reference would break Alpine builds.
static void call_malloc_trim(void) {
	void (*trim)(size_t) = (void (*)(size_t))dlsym(RTLD_DEFAULT, "malloc_trim");
	if (trim != NULL) {
		trim(0);
	}
}
*/
import "C"

// Release returns the free memory of the C heap to the OS. It is a no-op when
// the libc has no malloc_trim.
func Release() {
	C.call_malloc_trim()
}
