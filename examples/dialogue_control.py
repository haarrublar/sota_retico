import numpy as np
import retico_core
from retico_core.abstract import AbstractConsumingModule, AbstractModule, UpdateMessage
from retico_core.audio import AudioIU
from retico_core.text import SpeechRecognitionIU
from retico_speechbraintts import SpeechBrainTTSModule

states = {
    "mic_gate": True,
    "speaker_gate": True,
    "robot_talking": False,
}


def offon(gate_id: str, control: bool = True):
    if gate_id not in states:
        return print("Module state not found")
    else:
        states[gate_id] = control


class AudioAnnotator(AbstractModule):
    """Writes rms, seconds and (optionally) whose turn it is into each chunk's meta_data."""

    @staticmethod
    def name():
        return "Audio Annotator"

    @staticmethod
    def description():
        return "Tags each audio chunk with rms, seconds and turn"

    @staticmethod
    def input_ius():
        return [AudioIU]

    @staticmethod
    def output_iu():
        return AudioIU

    def __init__(self, state_id=None, **kwargs):
        super().__init__(**kwargs)
        if state_id is not None and state_id not in states:
            raise ValueError(f"Unknown {state_id}. Available: {list(states)}")
        self.state_id = state_id  # None = don't tag the turn

    def process_update(self, update_msg):
        turn = None
        if self.state_id is not None:
            turn = "robot" if states[self.state_id] else "user"  # read once per message
        for iu, ut in update_msg:
            x = np.frombuffer(iu.raw_audio, dtype=np.int16) / 32768
            iu.meta_data["rms"] = float(np.sqrt(np.mean(x**2))) if len(x) else 0.0
            iu.meta_data["seconds"] = len(iu.raw_audio) / (iu.sample_width * iu.rate)
            if turn is not None:
                iu.meta_data["turn"] = turn
        return update_msg


class AudioGate(AbstractModule):
    @staticmethod
    def name():
        return "Audio Gate"

    @staticmethod
    def description():
        return "Generate silence incremental units when opened"

    @staticmethod
    def input_ius():
        return [AudioIU]

    @staticmethod
    def output_iu():
        return AudioIU

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._is_talking = False

    def process_update(self, update_msg):
        out = UpdateMessage()
        for iu, ut in update_msg:
            if iu.meta_data.get("turn") == "robot":
                silent = self.create_iu(iu)
                silent.set_audio(
                    bytes(len(iu.raw_audio)), iu.nframes, iu.rate, iu.sample_width
                )
                out.add_iu(silent, ut)
            else:
                out.add_iu(iu, ut)
        return out


class RobotTalking(AbstractConsumingModule):
    """Robot turn: starts at the ASR commit, ends `tail` s after the TTS end marker."""

    @staticmethod
    def name():
        return "Robot Talking"

    @staticmethod
    def description():
        return "Sets a state for the robot's turn"

    @staticmethod
    def input_ius():
        return [AudioIU, SpeechRecognitionIU]  # audio from the tts, text from the asr

    @staticmethod
    def output_iu():
        return None

    def __init__(self, state_id, tail=0.5, **kwargs):
        super().__init__(**kwargs)
        if state_id not in states:
            raise ValueError(f"Unknown {state_id}. Available: {list(states)}")
        self.state_id = state_id
        self.tail = tail  # loop delay (~0.2 s) + room echo (~0.3 s)
        self.after_end = None  # seconds counted since the end marker

    def process_update(self, update_msg):
        for iu, ut in update_msg:
            if isinstance(iu, SpeechRecognitionIU):  # start: the user finished
                if ut == retico_core.UpdateType.COMMIT and not states[self.state_id]:
                    offon(self.state_id, True)
                    print("\n[robot_talking True (commit)]")

            elif isinstance(iu, AudioIU) and states[self.state_id]:  # tts audio
                if iu.meta_data.get("marker") == "end":  # last bag of the answer
                    self.after_end = 0.0
                elif self.after_end is not None:  # wait the tail
                    self.after_end += len(iu.raw_audio) / (iu.sample_width * iu.rate)
                    if self.after_end >= self.tail:
                        offon(self.state_id, False)
                        self.after_end = None
                        print("\n[robot_talking False (end marker)]")
        return None


class MarkedTTS(SpeechBrainTTSModule):
    """SpeechBrain TTS that labels the last audio chunk of each answer."""

    def append(self, update_message):
        n = len(self.audio_buffer)
        for iu, ut in update_message:
            if n and self.audio_pointer == n:  # the chunk just taken was the last one
                iu.meta_data["marker"] = "end"
        super().append(update_message)


class RobotTag(AbstractModule):
    """Sits between the TTS and the speaker: tags every chunk as robot and passes it on."""

    @staticmethod
    def name():
        return "Robot Tag"

    @staticmethod
    def description():
        return "Tags speaker audio as robot"

    @staticmethod
    def input_ius():
        return [AudioIU]

    @staticmethod
    def output_iu():
        return AudioIU

    def process_update(self, update_msg):
        for iu, ut in update_msg:
            iu.meta_data["source"] = "robot"
            iu.meta_data["sound"] = any(iu.raw_audio)  # False = silent chunk
        return update_msg


class MicFilter(AbstractModule):
    """Receives robot-tagged speaker audio and mic audio, in time order.
    Deletes mic audio while the robot is sounding (plus `hold` s)."""

    @staticmethod
    def name():
        return "Mic Filter"

    @staticmethod
    def description():
        return "Deletes mic audio while the speaker plays robot audio"

    @staticmethod
    def input_ius():
        return [AudioIU]

    @staticmethod
    def output_iu():
        return AudioIU

    def __init__(self, hold=0.6, **kwargs):
        super().__init__(**kwargs)
        self.hold = hold  # NEW: seconds to stay closed after the last robot sound
        self.hold_left = 0.0  # NEW

    def process_update(self, update_msg):
        out = UpdateMessage()
        for iu, ut in update_msg:
            if iu.meta_data.get("source") == "robot":  # from the speaker side
                seconds = len(iu.raw_audio) / (iu.sample_width * iu.rate)  # NEW
                if iu.meta_data["sound"]:
                    self.hold_left = self.hold  # NEW: robot sounding → reset the hold
                else:
                    self.hold_left = max(
                        0.0, self.hold_left - seconds
                    )  # NEW: count down
                continue  # not forwarded to the asr
            if self.hold_left > 0:  # mic chunk during robot sound (or hold)
                continue  # deleted
            iu.meta_data["source"] = "user"
            out.add_iu(iu, ut)
        return out if len(out) else None


# sample use
# gate = AudioGate("mic_gate")

# iu = gate.create_iu()
# iu.set_audio(
#     np.full(320, 1000, dtype=np.int16).tobytes(), 320, 16000, 2
# )
# msg = UpdateMessage.from_iu(iu, UpdateType.ADD)

# for state in (True, False):
#     offon("mic_gate", state)
#     out = gate.process_update(msg)
#     for x, ut in out:
#         samples = np.frombuffer(x.raw_audio, dtype=np.int16)
#         print(state, ut, len(x.raw_audio), samples.max())
