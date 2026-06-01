package server

import (
	"bytes"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestStreamParseBool(t *testing.T) {
	tests := []struct {
		input    string
		expected bool
	}{
		{"true", true},
		{"1", true},
		{"yes", true},
		{"on", true},
		{"false", false},
		{"0", false},
		{"", false},
		{"nope", false},
		{" TRUE ", true},
		{" Yes ", true},
	}

	for _, tc := range tests {
		t.Run(fmt.Sprintf("input_%q", tc.input), func(t *testing.T) {
			if got := parseBool(tc.input); got != tc.expected {
				t.Errorf("parseBool(%q) = %v; want %v", tc.input, got, tc.expected)
			}
		})
	}
}

func TestStreamSSEFormat(t *testing.T) {
	event := StreamDeltaEvent{
		Type:  "transcript.text.delta",
		Delta: "hola",
	}

	data, err := json.Marshal(event)
	if err != nil {
		t.Fatalf("failed to marshal event: %v", err)
	}

	expected := fmt.Sprintf("event: %s\ndata: %s\n\n", event.Type, string(data))

	var buf bytes.Buffer
	buf.WriteString("event: " + event.Type + "\n")
	buf.WriteString("data: " + string(data) + "\n\n")

	if buf.String() != expected {
		t.Errorf("Format mismatch. Got %q, want %q", buf.String(), expected)
	}
}

func TestStreamResponseRecorderImplementsFlusher(t *testing.T) {
	rec := httptest.NewRecorder()
	_, ok := interface{}(rec).(http.Flusher)
	if !ok {
		t.Skip("httptest.ResponseRecorder does not implement http.Flusher in this Go version")
	}
}

// sseEvent represents a single Server-Sent Event parsed from the stream.
type sseEvent struct {
	Event string
	Data  string
}

// parseSSEEvents is a test helper that extracts SSE events from a response body.
func parseSSEEvents(t *testing.T, body string) []sseEvent {
	t.Helper()
	var events []sseEvent

	// Events are separated by \n\n
	chunks := strings.Split(body, "\n\n")
	for _, chunk := range chunks {
		if strings.TrimSpace(chunk) == "" {
			continue
		}

		lines := strings.Split(chunk, "\n")
		var ev sseEvent
		for _, line := range lines {
			if strings.HasPrefix(line, "event: ") {
				ev.Event = strings.TrimPrefix(line, "event: ")
			} else if strings.HasPrefix(line, "data: ") {
				ev.Data = strings.TrimPrefix(line, "data: ")
			}
		}
		if ev.Event != "" || ev.Data != "" {
			events = append(events, ev)
		}
	}
	return events
}

func TestStreamSSEParserHelper(t *testing.T) {
	// Simulate SSE stream body with deltas and done event
	delta1 := StreamDeltaEvent{Type: "transcript.text.delta", Delta: "Hello "}
	delta2 := StreamDeltaEvent{Type: "transcript.text.delta", Delta: "world!"}
	done := StreamDoneEvent{Type: "transcript.text.done", Text: "Hello world!"}

	d1, _ := json.Marshal(delta1)
	d2, _ := json.Marshal(delta2)
	dd, _ := json.Marshal(done)

	body := fmt.Sprintf("event: transcript.text.delta\ndata: %s\n\n"+
		"event: transcript.text.delta\ndata: %s\n\n"+
		"event: transcript.text.done\ndata: %s\n\n",
		string(d1), string(d2), string(dd))

	events := parseSSEEvents(t, body)

	if len(events) != 3 {
		t.Fatalf("expected 3 events, got %d", len(events))
	}

	var fullText string
	for i, ev := range events {
		if i < 2 { // first two should be deltas
			if ev.Event != "transcript.text.delta" {
				t.Errorf("expected event type transcript.text.delta, got %q", ev.Event)
			}
			var d StreamDeltaEvent
			if err := json.Unmarshal([]byte(ev.Data), &d); err != nil {
				t.Fatalf("failed to unmarshal delta: %v", err)
			}
			fullText += d.Delta
		} else { // last should be done
			if ev.Event != "transcript.text.done" {
				t.Errorf("expected event type transcript.text.done, got %q", ev.Event)
			}
			var d StreamDoneEvent
			if err := json.Unmarshal([]byte(ev.Data), &d); err != nil {
				t.Fatalf("failed to unmarshal done: %v", err)
			}
			if d.Text != fullText {
				t.Errorf("done text mismatch: got %q, want %q", d.Text, fullText)
			}
		}
	}

	if fullText != "Hello world!" {
		t.Errorf("expected concatenated text 'Hello world!', got %q", fullText)
	}
}
