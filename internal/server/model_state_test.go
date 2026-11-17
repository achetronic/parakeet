// SPDX-FileCopyrightText: 2026 Alby Hernández <hola@achetronic.com>
// SPDX-License-Identifier: Apache-2.0

package server

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"parakeet/internal/asr"
)

// An unloaded transcriber is a valid state after an idle or manual unload, so a
// zero-value one exercises the read-only paths of the model endpoints without
// loading ONNX sessions.
func newUnloadedServer() *Server {
	return &Server{transcriber: &asr.Transcriber{}}
}

func TestHandleModelsReportsLoadedState(t *testing.T) {
	s := newUnloadedServer()

	rr := httptest.NewRecorder()
	s.handleModels(rr, httptest.NewRequest(http.MethodGet, "/v1/models", nil))

	if rr.Code != http.StatusOK {
		t.Fatalf("status = %d, want %d", rr.Code, http.StatusOK)
	}

	var resp ModelsResponse
	if err := json.Unmarshal(rr.Body.Bytes(), &resp); err != nil {
		t.Fatalf("decode response: %v", err)
	}
	if len(resp.Data) != 2 {
		t.Fatalf("got %d models, want 2", len(resp.Data))
	}
	for _, m := range resp.Data {
		if m.Loaded {
			t.Errorf("model %q reports loaded=true, want false for an unloaded transcriber", m.ID)
		}
	}
}

func TestHandleModelUnloadReportsTheStateAfterTheCall(t *testing.T) {
	s := newUnloadedServer()

	rr := httptest.NewRecorder()
	s.handleModelUnload(rr, httptest.NewRequest(http.MethodPost, "/v1/models/unload", nil))

	if rr.Code != http.StatusOK {
		t.Fatalf("status = %d, want %d", rr.Code, http.StatusOK)
	}

	var state ModelStateResponse
	if err := json.Unmarshal(rr.Body.Bytes(), &state); err != nil {
		t.Fatalf("decode response: %v", err)
	}
	if state.Object != "model.state" {
		t.Errorf("object = %q, want %q", state.Object, "model.state")
	}
	// Diarization is disabled here, so only the ASR model is reported.
	if len(state.Models) != 1 {
		t.Fatalf("got %d models in the answer, want 1", len(state.Models))
	}
	asr := state.Models[0]
	if asr.ID != modelID {
		t.Errorf("id = %q, want %q", asr.ID, modelID)
	}
	if asr.Loaded {
		t.Error("loaded = true, want false")
	}
}

func TestModelEndpointsRejectNonPost(t *testing.T) {
	handlers := map[string]http.HandlerFunc{
		"unload": newUnloadedServer().handleModelUnload,
		"load":   newUnloadedServer().handleModelLoad,
	}

	for name, handler := range handlers {
		t.Run(name, func(t *testing.T) {
			rr := httptest.NewRecorder()
			handler(rr, httptest.NewRequest(http.MethodGet, "/v1/models/"+name, nil))

			if rr.Code != http.StatusMethodNotAllowed {
				t.Fatalf("status = %d, want %d", rr.Code, http.StatusMethodNotAllowed)
			}
		})
	}
}
