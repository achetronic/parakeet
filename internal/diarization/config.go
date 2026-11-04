// SPDX-FileCopyrightText: 2026 Alby Hernández <hola@achetronic.com>
// SPDX-License-Identifier: Apache-2.0

package diarization

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
)

// Config holds the inference parameters of the Nemotron-3-Diarization model,
// loaded from config.json and preprocessor_config.json.
type Config struct {
	// Offline chunking, in encoder frames (80 ms each).
	ChunkLength              int
	ChunkLeftContext         int
	ChunkRightContext        int
	FIFOLength               int
	SpeakerCacheUpdatePeriod int

	// Speaker cache / streaming.
	SpeakerCacheLength                  int
	SpeakerCacheSilenceFramesPerSpeaker int
	PredictionScoreThreshold            float64
	LatestFramesScoreBoost              float64
	MinPositiveScoresRate               float64
	StrongBoostRate                     float64
	WeakBoostRate                       float64
	NumSpeakers                         int
	SubsamplingFactor                   int
	HiddenSize                          int

	// Feature extractor.
	FeatureSize int
	HopLength   int
	NFFT        int
	WinLength   int
	Preemphasis float64
	SampleRate  int
}

// loadConfig reads and merges config.json and preprocessor_config.json from the
// model directory. Fields the checkpoint does not declare fall back to the
// upstream defaults (chunk_left_context=0).
func loadConfig(dir string) (Config, error) {
	cfg := Config{ChunkLeftContext: 0}

	var raw struct {
		ChunkLength              int `json:"chunk_length"`
		ChunkRightContext        int `json:"chunk_right_context"`
		ChunkLeftContext         int `json:"chunk_left_context"`
		FIFOLength               int `json:"fifo_length"`
		SpeakerCacheUpdatePeriod int `json:"speaker_cache_update_period"`
		AudioConfig              struct {
			SubsamplingFactor int `json:"subsampling_factor"`
			HiddenSize        int `json:"hidden_size"`
		} `json:"audio_config"`
		HeadConfig struct {
			NumSpeakers       int `json:"num_speakers"`
			SubsamplingFactor int `json:"subsampling_factor"`
		} `json:"head_config"`
		StreamingConfig struct {
			FIFOLength                int     `json:"fifo_length"`
			SpeakerCacheUpdatePeriod  int     `json:"speaker_cache_update_period"`
			SpeakerCacheLength        int     `json:"speaker_cache_length"`
			SpeakerCacheSilenceFrames int     `json:"speaker_cache_silence_frames_per_speaker"`
			PredictionScoreThreshold  float64 `json:"prediction_score_threshold"`
			LatestFramesScoreBoost    float64 `json:"latest_frames_score_boost"`
			MinPositiveScoresRate     float64 `json:"min_positive_scores_rate"`
			StrongBoostRate           float64 `json:"strong_boost_rate"`
			WeakBoostRate             float64 `json:"weak_boost_rate"`
			NumSpeakers               int     `json:"num_speakers"`
			SubsamplingFactor         int     `json:"subsampling_factor"`
		} `json:"streaming_config"`
	}
	if err := readJSON(filepath.Join(dir, "config.json"), &raw); err != nil {
		return Config{}, err
	}

	cfg.ChunkLength = raw.ChunkLength
	cfg.ChunkRightContext = raw.ChunkRightContext
	if raw.ChunkLeftContext != 0 {
		cfg.ChunkLeftContext = raw.ChunkLeftContext
	}
	cfg.FIFOLength = raw.FIFOLength
	cfg.SpeakerCacheUpdatePeriod = raw.SpeakerCacheUpdatePeriod
	cfg.SubsamplingFactor = raw.AudioConfig.SubsamplingFactor
	cfg.HiddenSize = raw.AudioConfig.HiddenSize
	cfg.NumSpeakers = raw.HeadConfig.NumSpeakers

	cfg.SpeakerCacheLength = raw.StreamingConfig.SpeakerCacheLength
	cfg.SpeakerCacheSilenceFramesPerSpeaker = raw.StreamingConfig.SpeakerCacheSilenceFrames
	cfg.PredictionScoreThreshold = raw.StreamingConfig.PredictionScoreThreshold
	cfg.LatestFramesScoreBoost = raw.StreamingConfig.LatestFramesScoreBoost
	cfg.MinPositiveScoresRate = raw.StreamingConfig.MinPositiveScoresRate
	cfg.StrongBoostRate = raw.StreamingConfig.StrongBoostRate
	cfg.WeakBoostRate = raw.StreamingConfig.WeakBoostRate

	var pre struct {
		FeatureSize int     `json:"feature_size"`
		HopLength   int     `json:"hop_length"`
		NFFT        int     `json:"n_fft"`
		WinLength   int     `json:"win_length"`
		Preemphasis float64 `json:"preemphasis"`
		SampleRate  int     `json:"sampling_rate"`
	}
	if err := readJSON(filepath.Join(dir, "preprocessor_config.json"), &pre); err != nil {
		return Config{}, err
	}
	cfg.FeatureSize = pre.FeatureSize
	cfg.HopLength = pre.HopLength
	cfg.NFFT = pre.NFFT
	cfg.WinLength = pre.WinLength
	cfg.Preemphasis = pre.Preemphasis
	cfg.SampleRate = pre.SampleRate

	if err := cfg.validate(); err != nil {
		return Config{}, err
	}
	return cfg, nil
}

func readJSON(path string, dst any) error {
	data, err := os.ReadFile(path)
	if err != nil {
		return fmt.Errorf("read %s: %w", path, err)
	}
	if err := json.Unmarshal(data, dst); err != nil {
		return fmt.Errorf("parse %s: %w", path, err)
	}
	return nil
}

func (c Config) validate() error {
	if c.ChunkLength <= 0 || c.ChunkRightContext <= 0 || c.SubsamplingFactor <= 0 {
		return fmt.Errorf("invalid chunk/subsampling config: chunk_length=%d chunk_right_context=%d subsampling=%d",
			c.ChunkLength, c.ChunkRightContext, c.SubsamplingFactor)
	}
	if c.NumSpeakers <= 0 || c.HiddenSize <= 0 || c.FeatureSize <= 0 {
		return fmt.Errorf("invalid model config: num_speakers=%d hidden_size=%d feature_size=%d",
			c.NumSpeakers, c.HiddenSize, c.FeatureSize)
	}
	return nil
}

// frameSeconds returns the duration of one encoder frame in seconds.
func (c Config) frameSeconds() float64 {
	return float64(c.SubsamplingFactor) * float64(c.HopLength) / float64(c.SampleRate)
}
