// SPDX-FileCopyrightText: 2026 Alby Hernández <hola@achetronic.com>
// SPDX-License-Identifier: Apache-2.0

package asr

import (
	"context"
	"errors"
	"testing"
	"time"
)

// These tests exercise the worker gate (acquireWorker/releaseWorker) without
// the ONNX Runtime or model files: those helpers only touch decoderPool.

// newTestPool builds a Transcriber whose pool holds n ready workers.
func newTestPool(n int) *Transcriber {
	tc := &Transcriber{decoderPool: make(chan *decoderWorker, n)}
	for i := 0; i < n; i++ {
		tc.decoderPool <- &decoderWorker{}
	}
	return tc
}

func TestAcquireWorkerBoundsToPool(t *testing.T) {
	tc := newTestPool(1)
	ctx := context.Background()

	w, err := tc.acquireWorker(ctx)
	if err != nil {
		t.Fatalf("first acquire: %v", err)
	}

	// With the only worker held, a second acquire must block until release.
	acquired := make(chan struct{})
	go func() {
		got, err := tc.acquireWorker(ctx)
		if err != nil {
			t.Errorf("second acquire: %v", err)
			close(acquired)
			return
		}
		tc.releaseWorker(got)
		close(acquired)
	}()

	select {
	case <-acquired:
		t.Fatal("second acquire returned before the worker was released")
	case <-time.After(50 * time.Millisecond):
	}

	tc.releaseWorker(w)

	select {
	case <-acquired:
	case <-time.After(time.Second):
		t.Fatal("second acquire did not return after release")
	}
}

func TestAcquireWorkerCanceled(t *testing.T) {
	tc := newTestPool(0) // empty pool
	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	if _, err := tc.acquireWorker(ctx); !errors.Is(err, context.Canceled) {
		t.Fatalf("acquireWorker = %v, want context.Canceled", err)
	}
}

func TestReleaseWorkerAfterClose(t *testing.T) {
	tc := newTestPool(1)

	w, err := tc.acquireWorker(context.Background())
	if err != nil {
		t.Fatalf("acquire: %v", err)
	}
	close(tc.decoderPool)
	tc.releaseWorker(w) // must not panic
}
