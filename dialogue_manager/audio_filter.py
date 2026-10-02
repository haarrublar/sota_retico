import math

import numpy as np
import retico_core
from retico_core.abstract import AbstractModule, UpdateMessage
from retico_core.audio import AudioIU


class FrameDFT:
    """Shared toolbox: Hann window, DFT and inverse DFT from the formulas."""

    def __init__(self, N=512, H=256):
        self.N = N  # samples per frame
        self.H = H  # hop between frames
        # w[n] = 0.5 - 0.5*cos(2*pi*n/(N-1))
        self.w = np.array(
            [0.5 - 0.5 * math.cos(2 * math.pi * n / (N - 1)) for n in range(N)]
        )
        k = np.arange(N // 2 + 1)  # bins 0..N/2 (the rest is the mirror)
        n = np.arange(N)
        angle = 2 * np.pi * np.outer(k, n) / N  # angle[k, n] = 2*pi*k*n/N
        self.C = np.cos(angle)
        self.S = np.sin(angle)
        # inverse DFT weights: bins 1..N/2-1 appear twice (themselves + their mirror)
        self.a = np.full(N // 2 + 1, 2.0)
        self.a[0] = 1.0
        self.a[-1] = 1.0

    def dft(self, frame):
        """Re X[k] = sum_n x[n]*cos(2*pi*k*n/N),  Im X[k] = -sum_n x[n]*sin(2*pi*k*n/N)"""
        return self.C @ frame, -(self.S @ frame)

    def idft(self, re, im):
        """x[n] = (1/N) * sum_k a_k * (Re X[k]*cos(2*pi*k*n/N) - Im X[k]*sin(2*pi*k*n/N))"""
        return ((self.a * re) @ self.C - (self.a * im) @ self.S) / self.N

    def bins(self, freqs_hz, width_hz, rate):
        """Bin numbers k within ±width_hz of each frequency (bin k = k*rate/N Hz)."""
        k_hz = np.arange(self.N // 2 + 1) * rate / self.N
        return np.array(
            [
                k
                for k, f in enumerate(k_hz)
                if any(abs(f - c) <= width_hz for c in freqs_hz)
            ],
            dtype=int,
        )


class MarkDecoder(AbstractModule):
    """Finds Sota's mark. mode="remove": Sota → silence.  mode="keep": only Sota passes."""

    @staticmethod
    def name():
        return "Mark Decoder"

    @staticmethod
    def description():
        return "Detects the robot's frequency mark"

    @staticmethod
    def input_ius():
        return [AudioIU]

    @staticmethod
    def output_iu():
        return AudioIU

    def __init__(
        self, mode="remove", rate=16000, width_hz=150, threshold=0.5, hold=25, **kwargs
    ):
        super().__init__(**kwargs)
        self.t = FrameDFT()
        self.mode = mode
        self.threshold = threshold
        self.hold = (
            hold  # chunks to stay "Sota" after the mark was seen (25 × 20 ms = 0.5 s)
        )
        self.counter = hold  # chunks since the mark was last seen
        self.ratio = 1.0
        self.buffer = np.array([])
        self.mark = self.t.bins((2500, 3500), width_hz, rate)  # inside the notches
        ring = self.t.bins((2500, 3500), width_hz + 300, rate)
        self.near = np.setdiff1d(
            ring, self.t.bins((2500, 3500), width_hz + 60, rate)
        )  # just outside

    def process_update(self, update_message):
        out = UpdateMessage()
        for iu, ut in update_message:
            x = np.frombuffer(iu.raw_audio, dtype=np.int16) / 32768
            self.buffer = np.concatenate([self.buffer, x])
            while len(self.buffer) >= self.t.N:  # 1. ratio of every new frame
                frame = self.buffer[: self.t.N]
                if np.sqrt(np.mean(frame**2)) > 0.01:  # only frames with sound
                    re, im = self.t.dft(frame * self.t.w)
                    P = re**2 + im**2
                    new = P[self.mark].mean() / (P[self.near].mean() + 1e-12)
                    self.ratio = (
                        0.7 * self.ratio + 0.3 * new
                    )  # smooth: one odd frame can't decide
                self.buffer = self.buffer[self.t.H :]

            if self.ratio < self.threshold:  # 2. mark seen → restart the hold
                self.counter = 0
            else:
                self.counter += 1
            sota = self.counter < self.hold

            if self.mode == "keep" and sota:  # 3. route the chunk
                out.add_iu(iu, ut)
            elif self.mode == "remove":
                if sota:
                    silent = self.create_iu(iu)
                    silent.set_audio(
                        bytes(len(iu.raw_audio)), iu.nframes, iu.rate, iu.sample_width
                    )
                    out.add_iu(silent, ut)
                else:
                    out.add_iu(iu, ut)
        return out if len(out) else None


class MarkEncoder(AbstractModule):
    """Removes narrow frequency bands (the mark) from the audio: TTS → encoder → speaker."""

    @staticmethod
    def name():
        return "Mark Encoder"

    @staticmethod
    def description():
        return (
            "Cuts notches at fixed frequencies so the robot's voice can be recognized"
        )

    @staticmethod
    def input_ius():
        return [AudioIU]

    @staticmethod
    def output_iu():
        return AudioIU

    def __init__(self, mark_hz=(2500, 3500), width_hz=60, N=512, H=256, **kwargs):
        super().__init__(**kwargs)
        self.tools = FrameDFT(N, H)
        self.mark_hz = mark_hz  # centres of the notches
        self.width_hz = width_hz  # half-width of each notch
        self.mark_bins = None  # computed on the first chunk, when the rate is known
        self.inbuf = np.array([])  # samples waiting to be framed
        self.num = np.zeros(N)  # overlap-add: sum of windowed output frames
        self.den = np.zeros(N)  # overlap-add: sum of squared windows
        self.ready = np.array([])  # finished output samples

    def encode_samples(self, x, rate):
        """Feeds samples in, returns the samples that are finished (same timeline as the input)."""
        t = self.tools
        if self.mark_bins is None:
            self.mark_bins = t.bins(self.mark_hz, self.width_hz, rate)
        self.inbuf = np.concatenate([self.inbuf, x])
        while len(self.inbuf) >= t.N:
            frame = self.inbuf[: t.N] * t.w  # x_m[n] = x[m*H + n] * w[n]
            re, im = t.dft(frame)  # X_m[k]
            re[self.mark_bins] = 0  # X'[k] = 0 in the mark bins
            im[self.mark_bins] = 0
            y = t.idft(re, im)  # x'_m[n]
            self.num += y * t.w  # overlap-add, windowed again
            self.den += t.w**2
            done = np.where(
                self.den[: t.H] > 1e-3,
                self.num[: t.H] / np.maximum(self.den[: t.H], 1e-3),
                0.0,
            )
            self.ready = np.concatenate([self.ready, done])  # first H samples are final
            self.num = np.concatenate(
                [self.num[t.H :], np.zeros(t.H)]
            )  # shift to next frame
            self.den = np.concatenate([self.den[t.H :], np.zeros(t.H)])
            self.inbuf = self.inbuf[t.H :]
        out, self.ready = self.ready, np.array([])
        return out

    def process_update(self, update_message):
        out_msg = UpdateMessage()
        for iu, ut in update_message:
            if ut != retico_core.UpdateType.ADD:
                continue
            x = np.frombuffer(iu.raw_audio, dtype=np.int16) / 32768
            self.ready = np.concatenate([self.ready, self.encode_samples(x, iu.rate)])
            size = len(x)  # send chunks of the same size as the incoming ones
            while len(self.ready) >= size:
                chunk, self.ready = self.ready[:size], self.ready[size:]
                out_iu = self.create_iu(iu)
                data = (np.clip(chunk, -1, 1) * 32767).astype(np.int16).tobytes()
                out_iu.set_audio(data, size, iu.rate, 2)
                out_msg.add_iu(out_iu, ut)
        return out_msg if len(out_msg) else None
