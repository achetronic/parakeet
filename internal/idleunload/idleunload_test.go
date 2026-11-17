// SPDX-FileCopyrightText: 2026 Alby Hernández <hola@achetronic.com>
// SPDX-License-Identifier: Apache-2.0

package idleunload

import (
	"sync/atomic"
	"testing"
	"time"
)

// alwaysUnload records how many times the watcher asked to unload.
func alwaysUnload(calls *atomic.Int32) func(time.Duration) bool {
	return func(time.Duration) bool {
		calls.Add(1)
		return true
	}
}

func waitForCalls(t *testing.T, calls *atomic.Int32, want int32, within time.Duration) {
	t.Helper()
	deadline := time.Now().Add(within)
	for time.Now().Before(deadline) {
		if calls.Load() >= want {
			return
		}
		time.Sleep(5 * time.Millisecond)
	}
	t.Fatalf("got %d unload calls, want %d within %v", calls.Load(), want, within)
}

func TestWatcherUnloadsAfterTimeout(t *testing.T) {
	var calls atomic.Int32
	w := New("test", alwaysUnload(&calls))
	defer w.Stop()

	w.Start(40 * time.Millisecond)
	waitForCalls(t, &calls, 1, 2*time.Second)
}

func TestWatcherTouchPostponesTheUnload(t *testing.T) {
	var calls atomic.Int32
	w := New("test", alwaysUnload(&calls))
	defer w.Stop()

	const timeout = 150 * time.Millisecond
	w.Start(timeout)

	// Touch past the halfway point: the unload must be pushed back, so nothing
	// may have fired at the original deadline.
	time.Sleep(timeout * 6 / 10)
	w.Touch()
	time.Sleep(timeout * 6 / 10)
	if calls.Load() != 0 {
		t.Fatalf("got %d unload calls before the restarted timeout, want 0", calls.Load())
	}

	waitForCalls(t, &calls, 1, 2*time.Second)
}

func TestWatcherStopEndsTheWatcher(t *testing.T) {
	var calls atomic.Int32
	w := New("test", alwaysUnload(&calls))

	w.Start(20 * time.Millisecond)
	waitForCalls(t, &calls, 1, 2*time.Second)

	w.Stop()
	stopped := calls.Load()
	time.Sleep(100 * time.Millisecond)
	if got := calls.Load(); got != stopped {
		t.Fatalf("got %d unload calls after Stop, want the %d seen at Stop", got, stopped)
	}

	// Stopping twice must not panic: Close and Shutdown can both reach it.
	w.Stop()
}

func TestWatcherNonPositiveTimeoutStaysIdle(t *testing.T) {
	var calls atomic.Int32
	w := New("test", alwaysUnload(&calls))
	defer w.Stop()

	w.Start(0)
	w.Start(-time.Second)
	time.Sleep(80 * time.Millisecond)
	if got := calls.Load(); got != 0 {
		t.Fatalf("got %d unload calls with the watcher disabled, want 0", got)
	}
}

func TestWatcherIdleForIsZeroBeforeAnyActivity(t *testing.T) {
	w := New("test", func(time.Duration) bool { return false })
	defer w.Stop()

	// Never touched means "not idle": an owner that was never used has nothing
	// to free, so a zero here keeps unloadIfIdle from firing.
	if got := w.IdleFor(); got != 0 {
		t.Fatalf("IdleFor() = %v before any activity, want 0", got)
	}

	w.Touch()
	if got := w.IdleFor(); got <= 0 {
		t.Fatalf("IdleFor() = %v right after Touch, want a positive age", got)
	}
}
