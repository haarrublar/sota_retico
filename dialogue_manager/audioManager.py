import wave

import numpy as np
import retico_core
import webrtcvad
from retico_core.abstract import AbstractModule, UpdateMessage
from retico_core.audio import AudioIU

# bufferSize': 640, 'sampleRate': 16000, 'sampleSize_bits': 16, 'channels': 1
# 2^16 = 65,536, and as it's centered then we take half −32768 … 0 … +32767
# VAD is better for recognizing human/robot audio (boolean)


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

    def __init__(self, vad_mode=1, **kwargs):
        """setting the vad_mode value in 2 is not accurate for expression like yey or happines expressions"""
        super().__init__(**kwargs)
        self.vad = webrtcvad.Vad(vad_mode)

    def process_update(self, update_msg):
        for iu, ut in update_msg:
            # print("annotator got owner:", iu.meta_data.get("owner"))
            x = np.frombuffer(iu.raw_audio, dtype=np.int16) / 32768  # normalization
            iu.meta_data["rms"] = float(np.sqrt(np.mean(x**2))) if len(x) else 0.0
            iu.meta_data["seconds"] = len(iu.raw_audio) / (iu.sample_width * iu.rate)
            iu.meta_data["speech"] = self.vad.is_speech(iu.raw_audio, iu.rate)
        return update_msg


class AudioClassifier(AbstractModule):
    """Passes only the chunks of one type (user / robot)."""

    @staticmethod
    def name():
        return "Audio Classifier"

    @staticmethod
    def description():
        return "Keeps the audio of one type, removes the rest"

    @staticmethod
    def input_ius():
        return [AudioIU]

    @staticmethod
    def output_iu():
        return AudioIU

    def __init__(self, keep, **kwargs):
        super().__init__(**kwargs)
        self.keep = keep

    def process_update(self, update_message):
        out = UpdateMessage()
        for iu, ut in update_message:
            if (
                ut == retico_core.UpdateType.ADD
                and iu.meta_data.get("owner") == self.keep
            ):
                out.add_iu(iu, ut)
        return out if len(out) else None


class SilenceRemover(AbstractModule):
    """Passes only chunks marked as speech"""

    @staticmethod
    def name():
        return "Audio with no silences"

    @staticmethod
    def description():
        return "Audio with content"

    @staticmethod
    def input_ius():
        return [AudioIU]

    @staticmethod
    def output_iu():
        return AudioIU

    def __init__(self, hold=15, **kwargs):
        """ "Each chunk has 0.2s so for not losing info within words we extend that to 0.3s of silence"""
        super().__init__(**kwargs)
        self.hold = hold
        self.counter = hold

    def process_update(self, update_message):
        out = UpdateMessage()
        for iu, ut in update_message:
            if ut != retico_core.UpdateType.ADD:
                continue
            if iu.meta_data.get("speech"):
                self.counter = 0  # speech: reset
            else:
                self.counter += 1  # no speech: one more in a row
            if self.counter < self.hold:
                out.add_iu(iu, ut)
        return out if len(out) else None


class WavFile(AbstractModule):
    """Generates a Wav file"""

    @staticmethod
    def name():
        return "WWav file generator"

    @staticmethod
    def description():
        return ""

    @staticmethod
    def input_ius():
        return [AudioIU]

    @staticmethod
    def output_iu():
        return None

    def __init__(self, path="output.wav", **kwargs):
        super().__init__(**kwargs)
        self.path = path
        self.wav_file = None

    def process_update(self, update_message):
        for x, ut in update_message:
            if ut != retico_core.UpdateType.ADD:
                continue
            if self.wav_file is None:  # first chunk: open once
                self.wav_file = wave.open(self.path, "wb")
                self.wav_file.setnchannels(1)
                self.wav_file.setsampwidth(x.sample_width)
                self.wav_file.setframerate(x.rate)
            self.wav_file.writeframes(x.raw_audio)
        return None

    def shutdown(self):
        print(f"shutdown {self.path}: open={self.wav_file is not None}")
        if self.wav_file is not None:
            self.wav_file.close()
            self.wav_file = None
