# Diarization test audio

`2-speakers.wav` is a synthetic two-speaker "conversation" for verifying speaker
diarization. It alternates two distinct voices so the diarizer must tell them
apart and keep each speaker's identity across the whole file (which spans
multiple encoder chunks).

- Duration: ~57 s, 16 kHz mono 16-bit PCM.
- Speaker A (0-15 s and 28.7-43.7 s): excerpt of `testdata/reference/learn-case-interviews.mp3` (single narrator).
- Speaker B (15-28.7 s and 43.7-57.4 s): LibriSpeech sample (`https://cdn-media.huggingface.co/speech_samples/sample1.flac`).

Expected diarization: two speakers, with the segments alternating A, B, A, B.

Regenerate:

```bash
ffmpeg -v error -i testdata/reference/learn-case-interviews.mp3 -t 15 -ar 16000 -ac 1 -y /tmp/A.wav
ffmpeg -v error -i https://cdn-media.huggingface.co/speech_samples/sample1.flac -ar 16000 -ac 1 -y /tmp/B.wav
ffmpeg -v error -i /tmp/A.wav -i /tmp/B.wav -i /tmp/A.wav -i /tmp/B.wav \
  -filter_complex "[0][1][2][3]concat=n=4:v=0:a=1" -y 2-speakers.wav
```

Reference diarization result (Nemotron-3-Diarization int8, CPU):

```
spk=0  0.65- 7.48
spk=0  8.33-11.46
spk=0 11.79-14.18
spk=1 15.16-21.55
spk=1 22.35-28.53
spk=0 29.35-36.17
spk=0 37.03-40.19
spk=0 40.47-42.88
spk=1 43.84-50.26
spk=1 51.03-57.36
```
