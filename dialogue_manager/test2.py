import wave

import numpy as np
from audio_filter import FrameDFT


def load(path):
    with wave.open(path) as f:
        assert f.getsampwidth() == 2 and f.getnchannels() == 1, "needs 16-bit mono"
        x = np.frombuffer(f.readframes(f.getnframes()), dtype=np.int16) / 32768
        return x, f.getframerate()


tts, tts_rate = load("tts_marked.wav")  # what Sota said (clean)
air, air_rate = load("air.wav")  # what the mic heard

t = FrameDFT()
mark = t.bins((2500, 3500), 60, air_rate)
near = np.setdiff1d(
    t.bins((2500, 3500), 300, air_rate), t.bins((2500, 3500), 120, air_rate)
)


def rms(x):
    return float(np.sqrt(np.mean(x**2))) if len(x) else 0.0


def ratio(x):
    values = []
    for i in range(0, len(x) - t.N, t.H):
        re, im = t.dft(x[i : i + t.N] * t.w)
        P = re**2 + im**2
        values.append(P[mark].mean() / (P[near].mean() + 1e-12))
    return float(np.median(values)) if values else float("nan")


print(
    f"tts_marked.wav: {len(tts) / tts_rate:.1f} s   air.wav: {len(air) / air_rate:.1f} s"
)
print("  time   sota speaking   air rms   air ratio")
for s in np.arange(0, min(len(tts) / tts_rate, len(air) / air_rate) - 0.5, 0.5):
    tts_block = tts[int(s * tts_rate) : int((s + 0.5) * tts_rate)]
    air_block = air[int(s * air_rate) : int((s + 0.5) * air_rate)]
    speaking = "yes" if rms(tts_block) > 0.01 else "   "
    print(f"{s:6.1f}   {speaking:13}   {rms(air_block):.3f}     {ratio(air_block):.2f}")
