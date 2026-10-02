import time

import numpy as np
from retico_core import AbstractModule, UpdateMessage
from retico_core.audio import AudioIU


class SpeakerGatingModule(AbstractModule):
    """Closes when the speaker gets audio; reopens once the mic hears Sota finish."""

    @staticmethod
    def name():
        return "Speaker Gating Module"

    @staticmethod
    def description():
        return (
            "Suppresses mic audio from speaker activity until the room is quiet again"
        )

    @staticmethod
    def input_ius():
        return [AudioIU]

    @staticmethod
    def output_iu():
        return AudioIU

    def __init__(
        self,
        speaker_module,
        sound_rms=0.02,  # mic louder than this = sound in the room
        quiet_rms=0.01,  # mic quieter than this = room is quiet
        hold=0.8,  # quiet time needed before reopening
        timeout=6.0,  # give up waiting if Sota is never heard
        on_open=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.speaker = speaker_module
        self.sound_rms = sound_rms
        self.quiet_rms = quiet_rms
        self.hold = hold
        self.timeout = timeout
        self.state = "OPEN"  # OPEN → WAITING → HEARING → OPEN
        self.since = 0.0
        self.last_sound = 0.0
        self.on_open = on_open

    def set_state(self, state, now):
        self.state = state
        self.since = now
        print(f"[GATE] {state}  t={now:.2f}")

    def process_update(self, update_message):
        out = UpdateMessage()
        for iu, ut in update_message:
            now = time.time()
            x = np.frombuffer(iu.raw_audio, dtype=np.int16) / 32768
            rms = float(np.sqrt(np.mean(x**2))) if len(x) else 0.0
            speaker_active = self.speaker.busy_until > now

            if self.state == "OPEN" and speaker_active:
                self.set_state("WAITING", now)  # Sota is about to speak

            elif self.state == "WAITING":
                if rms > self.sound_rms:
                    self.last_sound = now
                    self.set_state("HEARING", now)  # Sota's voice reached the mic
                elif not speaker_active and now - self.since > self.timeout:
                    self.set_state("OPEN", now)  # never heard it, don't stay stuck

            elif self.state == "HEARING":
                if rms > self.quiet_rms:
                    self.last_sound = now
                quiet_long_enough = now - self.last_sound > self.hold
                if not speaker_active and quiet_long_enough:
                    self.set_state("OPEN", now)

            if self.state == "OPEN":
                output_iu = self.create_iu(iu)
                output_iu.set_audio(iu.raw_audio, iu.nframes, iu.rate, iu.sample_width)
                out.add_iu(output_iu, ut)

        if len(out):
            self.append(out)
