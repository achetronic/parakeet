# Diarization test audio

`2-speakers.wav` is a synthetic two-speaker "conversation" for verifying speaker
diarization. It alternates two distinct voices with short silence gaps, so the
diarizer must tell them apart and keep each speaker's identity across the whole
file (which spans multiple encoder chunks).

- Duration: ~59 s, 16 kHz mono 16-bit PCM.
- Speaker A (0-15 s and 30-45 s): excerpt of `testdata/reference/learn-case-interviews.mp3` (single narrator).
- Speaker B (15-29.4 s and 45.4-59.2 s): LibriSpeech sample (`https://cdn-media.huggingface.co/speech_samples/sample1.flac`).
- 0.6 s silence gaps between turns.

Expected diarization: two speakers, four turns alternating A, B, A, B.

Regenerate:

```bash
ffmpeg -v error -i testdata/reference/learn-case-interviews.mp3 -t 15 -ar 16000 -ac 1 -y /tmp/A.wav
ffmpeg -v error -i https://cdn-media.huggingface.co/speech_samples/sample1.flac -ar 16000 -ac 1 -y /tmp/B.wav
ffmpeg -v error -f lavfi -i anullsrc=r=16000:cl=mono -t 0.6 -y /tmp/sil.wav
ffmpeg -v error -i /tmp/A.wav -i /tmp/sil.wav -i /tmp/B.wav -i /tmp/sil.wav \
  -i /tmp/A.wav -i /tmp/sil.wav -i /tmp/B.wav \
  -filter_complex "[0][1][2][3][4][5][6]concat=n=7:v=0:a=1" -y 2-speakers.wav
```

Reference result (diarize=true, Nemotron-3-Diarization int8, CPU):

```
spk=0  0.56-14.80  In this video, I'm gonna teach you fundamental strategies...
spk=1 15.68-29.68  Going along slushy country roads and speaking to damp audiences...
spk=0 30.40-44.56  In this video, I'm gonna teach you fundamental strategies...
spk=1 45.60-59.04  Going along slushy country roads and speaking to damp audiences...
```
