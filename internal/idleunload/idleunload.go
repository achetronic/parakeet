// SPDX-FileCopyrightText: 2026 Alby Hernández <hola@achetronic.com>
// SPDX-License-Identifier: Apache-2.0

// Package idleunload keeps the "free the weights after a quiet period" policy
// out of the model holders. The holder records activity and provides a way to
// unload; this package owns the timer and nothing else.
package idleunload

import (
	"log/slog"
	"sync"
	"sync/atomic"
	"time"
)

// Watcher tracks the activity of one model holder.
type Watcher struct {
	// name identifies the model holder in the logs.
	name string
	// unloadIfIdle unloads only when the holder confirms, under its own lock,
	// that the whole timeout elapsed. It reports whether it unloaded.
	unloadIfIdle func(timeout time.Duration) bool

	notify   chan struct{}
	stop     chan struct{}
	stopOnce sync.Once
	last     atomic.Int64 // unix nanoseconds of the last recorded activity
}

// New returns a watcher for the named holder that asks unloadIfIdle to unload
// whenever the model may have been idle long enough.
func New(name string, unloadIfIdle func(timeout time.Duration) bool) *Watcher {
	return &Watcher{
		name:         name,
		unloadIfIdle: unloadIfIdle,
		notify:       make(chan struct{}, 1),
		stop:         make(chan struct{}),
	}
}

// Start launches the watcher. A non-positive timeout disables it.
func (w *Watcher) Start(timeout time.Duration) {
	if timeout <= 0 {
		return
	}
	slog.Info("idle unloading enabled", "model", w.name, "timeout", timeout)
	w.Touch()
	go w.watch(timeout)
}

// Touch records activity and wakes the watcher. It never blocks: a
// notification already pending resets the timer anyway, so dropping the extra
// one loses nothing.
func (w *Watcher) Touch() {
	w.last.Store(time.Now().UnixNano())
	select {
	case w.notify <- struct{}{}:
	default:
	}
}

// IdleFor reports how long ago the last activity was. Zero means none was ever
// recorded, so the holder is not idle: there is nothing to free yet.
func (w *Watcher) IdleFor() time.Duration {
	last := w.last.Load()
	if last == 0 {
		return 0
	}
	return time.Since(time.Unix(0, last))
}

// Stop ends the watcher. Safe to call more than once.
func (w *Watcher) Stop() {
	w.stopOnce.Do(func() { close(w.stop) })
}

func (w *Watcher) watch(timeout time.Duration) {
	timer := time.NewTimer(timeout)
	defer timer.Stop()
	for {
		select {
		case <-w.stop:
			return
		case <-w.notify:
			// Drain a timer that fired while this case was selected, then
			// restart it, so the timeout counts from the last activity.
			if !timer.Stop() {
				select {
				case <-timer.C:
				default:
				}
			}
			timer.Reset(timeout)
		case <-timer.C:
			if w.unloadIfIdle(timeout) {
				timer.Reset(timeout)
				continue
			}
			// Either a request is still running or one just finished: wait out
			// whatever is left of the timeout instead of spinning.
			if remaining := timeout - w.IdleFor(); remaining > 0 {
				timer.Reset(remaining)
			} else {
				timer.Reset(timeout)
			}
		}
	}
}
