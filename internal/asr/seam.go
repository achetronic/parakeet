// SPDX-FileCopyrightText: 2026 Alby Hernández <hola@achetronic.com>
// SPDX-License-Identifier: Apache-2.0

package asr

// This file implements seam-level token deduplication for long-audio chunking.
// Even with the boundary placed on silence, the two windows straddling a seam
// timestamp a word landing near the boundary slightly differently, so a token
// can still be emitted twice (duplicate) or, less often, as two different
// guesses at the same position. Dedup runs on the emitted tokens tagged with
// their ABSOLUTE encoder-frame timesteps and is always on (no flag).

const (
	// seamMaxTokens caps how many leading tokens of a window are held and
	// compared against the previous window's tail. k <= 3 keeps the streaming
	// buffer tiny: only the first few tokens of each window (after the first)
	// wait for the seam check, everything else streams as it is decoded.
	seamMaxTokens = 3

	// seamTimestepToleranceFrames is how close two tokens must be, in absolute
	// encoder frames (80 ms each), to count as the same acoustic position at a
	// seam. 3 frames is ~240 ms, wide enough to absorb the per-window timestamp
	// jitter that makes a boundary-straddling word land a frame or two apart in
	// each window, narrow enough not to swallow genuinely distinct neighbours.
	seamTimestepToleranceFrames = 3
)

// decodedToken is a token emitted by the TDT decoder tagged with its ABSOLUTE
// encoder-frame timestep. Absolute timesteps (as opposed to per-window local
// ones) let dedupSeam line up tokens emitted by two different windows that cover
// the same audio around a seam.
type decodedToken struct {
	id       int
	timestep int64
}

// dedupSeam decides which of window i+1's leading tokens (head) survive when
// compared against window i's trailing tokens (prevTail). It returns the
// survivors in order.
//
// A head token is dropped when its timestep is within seamTimestepToleranceFrames
// of any of the previous window's last seamMaxTokens tokens. That single rule
// covers the two failure modes from issue #18:
//
//   - Same text at (nearly) the same timestep: a duplicate ("to to"); drop it.
//   - Different text at (nearly) the same timestep: a collision; window i wins
//     because its LSTM reaches the seam fully warmed up while window i+1 is still
//     warming up, so drop window i+1's guess.
//
// A duplicate whose timesteps are further apart than the tolerance is kept, as
// are tokens that do not collide with the previous tail at all.
func dedupSeam(prevTail, head []decodedToken) []decodedToken {
	if len(prevTail) == 0 || len(head) == 0 {
		return head
	}

	// Only the tail-most tokens of the previous window can sit at the seam.
	tail := prevTail
	if len(tail) > seamMaxTokens {
		tail = tail[len(tail)-seamMaxTokens:]
	}

	survivors := make([]decodedToken, 0, len(head))
	for _, h := range head {
		drop := false
		for _, p := range tail {
			if absDiffInt64(h.timestep, p.timestep) <= seamTimestepToleranceFrames {
				drop = true
				break
			}
		}
		if !drop {
			survivors = append(survivors, h)
		}
	}
	return survivors
}

func absDiffInt64(a, b int64) int64 {
	if a > b {
		return a - b
	}
	return b - a
}
