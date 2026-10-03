import queue
import time
from collections import deque

import numpy as np
from retico_core import AbstractModule, UpdateMessage
from retico_core.audio import AudioIU
from timeline import mark


class AudioGatingModule(AbstractModule):
    """Closes on the user's commit, reopens once sota's voice has played and gone quiet."""

    SPEECH_RMS = 0.03  # mic above this = voice
    NOISE_SUM = 0.15  # window sum below this = silence
    WINDOW = 30  # 30 x 20 ms = 0.6 s, longer than sota's pauses
    LOUD_CHUNKS = 5  # 0.1 s of voice before we trust it's sota

    def __init__(self, speaker_module, **kwargs):
        super().__init__(**kwargs)
        self.speaker = speaker_module
        self._window = deque(maxlen=self.WINDOW)
        self._is_open = True
        self._closed_at = 0.0
        self._reset_turn()

    @staticmethod
    def name():
        return "Turn Gating Module"

    @staticmethod
    def description():
        return "User -> sota -> user, based on mic RMS"

    @staticmethod
    def input_ius():
        return [AudioIU]

    @staticmethod
    def output_iu():
        return AudioIU

    def _reset_turn(self):
        self._base = self.speaker.sent_seconds  # speech sent before this turn
        self._sent = False  # speaker started sending this turn
        self._loud = 0  # loud chunks since then
        self._first_heard = 0.0  # when sota's voice reached the mic
        self._window.clear()

    def expect_reply(self, *_):
        """Call on ASR commit: user is done, sota's turn starts."""
        self._reset_turn()
        self._is_open = False
        self._closed_at = time.time()
        print(f"gate: CLOSED (sota's turn)  t={self._closed_at:.2f}")

    def _update(self, rms, now):
        self._window.append(rms)

        if self.speaker.speaking:
            self._sent = True
        if self._sent and rms > self.SPEECH_RMS:
            self._loud += 1
            if self._loud == self.LOUD_CHUNKS:
                self._first_heard = now

        heard = self._loud >= self.LOUD_CHUNKS
        sent = self.speaker.sent_seconds - self._base
        played_all = heard and now > self._first_heard + sent
        silence = (
            len(self._window) == self.WINDOW and sum(self._window) < self.NOISE_SUM
        )

        if played_all and silence and not self.speaker.speaking:
            self._is_open = True
            print(f"gate: OPEN (your turn)  t={now:.2f}")
            print(f"  closed for   {now - self._closed_at:.2f} s")
            print(
                f"  sota started {self._first_heard - self._closed_at:.2f} s after close"
            )
            print(f"  speech sent  {sent:.2f} s")
            print(f"  extra wait   {now - self._first_heard - sent:.2f} s")

    def process_update(self, update_message):
        messages = [update_message]
        for q in self._left_buffers:
            while True:
                try:
                    messages.append(q.get_nowait())
                except queue.Empty:
                    break

        out = UpdateMessage()
        t_start = time.perf_counter()
        for iu, ut in update_message:
            mark("gate_lag", time.time() - iu.created_at)
            if not self._is_open:
                x = np.frombuffer(iu.raw_audio, dtype=np.int16) / 32768
                rms = float(np.sqrt(np.mean(x**2))) if len(x) else 0.0
                self._update(rms, time.time())

            if self._is_open:
                o = self.create_iu(iu)
                o.set_audio(iu.raw_audio, iu.nframes, iu.rate, iu.sample_width)
                out.add_iu(o, ut)

        if len(out):
            self.append(out)
        self._calls = getattr(self, "_calls", 0) + 1
        if self._calls % 50 == 0:  # once per second
            took = (time.perf_counter() - t_start) * 1000
            waiting = sum(q.qsize() for q in self._left_buffers)
            print(f"[GATE] process_update {took:.1f} ms, chunks waiting {waiting}")
