import sys
import time
import wave

import numpy as np
from transformers import WhisperForConditionalGeneration, WhisperProcessor

WAV = sys.argv[1]
MODELS = [
    "openai/whisper-tiny.en",
    "openai/whisper-base.en",
    "openai/whisper-base",
    "distil-whisper/distil-small.en",
]
RUNS = 5

from scipy.signal import resample_poly

with wave.open(WAV) as w:
    rate = w.getframerate()
    channels = w.getnchannels()
    audio = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16) / 32768

if channels == 2:  # stereo → mono
    audio = audio.reshape(-1, 2).mean(axis=1)
if rate != 16000:  # e.g. 48000 → 16000
    audio = resample_poly(audio, 16000, rate)

seconds = len(audio) / 16000
print(f"audio: {seconds:.2f}s (original {rate} Hz, {channels} ch)\n")


for name in MODELS:
    proc = WhisperProcessor.from_pretrained(name)
    model = WhisperForConditionalGeneration.from_pretrained(name)
    feats = proc(audio, sampling_rate=16000, return_tensors="pt").input_features

    model.generate(feats)  # warm-up, the first run is always slower
    times = []
    for _ in range(RUNS):
        t = time.perf_counter()
        ids = model.generate(feats)
        times.append(time.perf_counter() - t)

    text = proc.batch_decode(ids, skip_special_tokens=True)[0]
    mean = np.mean(times)
    print(f"{name}")
    print(f"  time: {mean:.2f}s ± {np.std(times):.2f}   RTF: {mean / seconds:.2f}")
    print(f"  text: {text}\n")
