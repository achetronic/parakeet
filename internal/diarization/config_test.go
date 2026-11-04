// SPDX-FileCopyrightText: 2026 Alby Hernández <hola@achetronic.com>
// SPDX-License-Identifier: Apache-2.0

package diarization

import (
	"os"
	"path/filepath"
	"testing"
)

func TestLoadConfig(t *testing.T) {
	dir := t.TempDir()

	configJSON := `{
		"chunk_length": 340,
		"chunk_right_context": 40,
		"fifo_length": 40,
		"speaker_cache_update_period": 300,
		"audio_config": {"subsampling_factor": 8, "hidden_size": 512},
		"head_config": {"num_speakers": 8, "subsampling_factor": 8},
		"streaming_config": {
			"fifo_length": 264,
			"speaker_cache_update_period": 222,
			"speaker_cache_length": 264,
			"speaker_cache_silence_frames_per_speaker": 1,
			"prediction_score_threshold": 0.25,
			"latest_frames_score_boost": 0.05,
			"min_positive_scores_rate": 0.5,
			"strong_boost_rate": 0.75,
			"weak_boost_rate": 1.5,
			"num_speakers": 8,
			"subsampling_factor": 8
		}
	}`
	preJSON := `{
		"feature_size": 128,
		"hop_length": 160,
		"n_fft": 512,
		"win_length": 400,
		"preemphasis": 0.97,
		"sampling_rate": 16000
	}`

	if err := os.WriteFile(filepath.Join(dir, "config.json"), []byte(configJSON), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "preprocessor_config.json"), []byte(preJSON), 0o644); err != nil {
		t.Fatal(err)
	}

	cfg, err := loadConfig(dir)
	if err != nil {
		t.Fatalf("loadConfig: %v", err)
	}

	if cfg.ChunkLength != 340 || cfg.ChunkRightContext != 40 || cfg.ChunkLeftContext != 0 {
		t.Errorf("chunk config wrong: %+v", cfg)
	}
	if cfg.SpeakerCacheLength != 264 || cfg.NumSpeakers != 8 || cfg.HiddenSize != 512 {
		t.Errorf("cache/model config wrong: %+v", cfg)
	}
	if cfg.Preemphasis != 0.97 || cfg.FeatureSize != 128 || cfg.HopLength != 160 {
		t.Errorf("feature config wrong: %+v", cfg)
	}
}

func TestConfigValidateRejectsZero(t *testing.T) {
	if err := (Config{}).validate(); err == nil {
		t.Fatal("expected validation error for zero config")
	}
}
