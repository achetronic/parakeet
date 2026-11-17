// SPDX-FileCopyrightText: 2026 Alby Hernández <hola@achetronic.com>
// SPDX-License-Identifier: Apache-2.0

package asr

import (
	"context"
	"testing"
	"time"

	"parakeet/internal/idleunload"
)

// These tests exercise the load/unload state machine without the ONNX Runtime
// or the model files: a transcriber marked as loaded with a pool of empty
// workers is enough, because destroy() is nil-safe and nothing runs inference.

// newLoadedTestTranscriber returns a transcriber that believes its sessions are
// in memory, with n pool workers and the idle watcher wired.
func newLoadedTestTranscriber(n int) *Transcriber {
	tc := newTestPool(n)
	tc.idle = idleunload.New("test", tc.unloadIfIdle)
	tc.loaded = true
	return tc
}

func TestUnloadFreesTheSessionsAndIsIdempotent(t *testing.T) {
	tc := newLoadedTestTranscriber(1)

	if !tc.Loaded() {
		t.Fatal("Loaded() = false, want true before the first unload")
	}
	tc.Unload()
	if tc.Loaded() {
		t.Fatal("Loaded() = true, want false after Unload")
	}
	if tc.decoderPool != nil {
		t.Fatal("decoderPool is not nil after Unload: it would be reused in a closed state")
	}

	// Unloading again must be a harmless no-op, not a panic on the closed pool.
	tc.Unload()
	if tc.Loaded() {
		t.Fatal("Loaded() = true after a second Unload, want false")
	}
}

func TestUnloadWaitsForInFlightRequests(t *testing.T) {
	tc := newLoadedTestTranscriber(1)
	ctx := context.Background()

	release, err := tc.acquireSessions(ctx)
	if err != nil {
		t.Fatalf("acquireSessions: %v", err)
	}

	done := make(chan struct{}, 1)
	go func() {
		tc.Unload()
		done <- struct{}{}
	}()

	select {
	case <-done:
		t.Fatal("Unload returned while a request still held the sessions")
	case <-time.After(50 * time.Millisecond):
	}

	release()

	select {
	case <-done:
	case <-time.After(2 * time.Second):
		t.Fatal("Unload did not return after the request released the sessions")
	}
	if tc.Loaded() {
		t.Fatal("Loaded() = true after Unload, want false")
	}
}

func TestIdleUnloadFreesWhenIdle(t *testing.T) {
	tc := newLoadedTestTranscriber(1)
	defer tc.Close()

	const timeout = 60 * time.Millisecond
	tc.StartIdleUnload(timeout)

	if !waitUntilUnloaded(tc, 2*time.Second) {
		t.Fatal("model still loaded after the idle timeout")
	}
}

func TestIdleUnloadRestartsOnActivity(t *testing.T) {
	tc := newLoadedTestTranscriber(1)
	defer tc.Close()

	const timeout = 200 * time.Millisecond
	tc.StartIdleUnload(timeout)

	// Touch at 60% of the timeout: the unload must move out past it.
	time.Sleep(timeout * 6 / 10)
	tc.touch()
	time.Sleep(timeout * 6 / 10)
	if !tc.Loaded() {
		t.Fatal("model unloaded despite activity inside the idle timeout")
	}

	if !waitUntilUnloaded(tc, 2*time.Second) {
		t.Fatal("model still loaded well past the idle timeout")
	}
}

func TestIdleUnloadDisabledByNonPositiveTimeout(t *testing.T) {
	tc := newLoadedTestTranscriber(1)
	defer tc.Close()

	tc.StartIdleUnload(0)
	time.Sleep(80 * time.Millisecond)
	if !tc.Loaded() {
		t.Fatal("model unloaded with the idle unloader disabled")
	}
}

// waitUntilUnloaded polls until the model reports unloaded or the deadline
// passes, which keeps these tests off a fixed sleep.
func waitUntilUnloaded(tc *Transcriber, within time.Duration) bool {
	deadline := time.Now().Add(within)
	for time.Now().Before(deadline) {
		if !tc.Loaded() {
			return true
		}
		time.Sleep(5 * time.Millisecond)
	}
	return !tc.Loaded()
}
