// SPDX-FileCopyrightText: 2026 Alby Hernández <hola@achetronic.com>
// SPDX-License-Identifier: Apache-2.0

// Package memtrim hands free C heap memory back to the operating system.
//
// ONNX Runtime allocates the model weights and its workspaces with malloc.
// Destroying a session frees them, but the libc allocator keeps the freed
// chunks inside its arenas, so the process RSS does not drop and a co-tenant on
// the same host gains nothing. Release() asks glibc to return them.
package memtrim
