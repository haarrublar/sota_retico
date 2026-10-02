import hashlib
import os
import threading
import time

import retico_core
from piper import PiperVoice

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VOICE_PATH = os.path.join(BASE_DIR, "en_US-lessac-medium.onnx")


class PiperTTS:
    def __init__(
        self,
        voice_path="en_US-lessac-medium.onnx",
        tmp_dir="~/.cache/piper_tts",
        caching=True,
        use_cuda=False,
    ):
        self.voice = PiperVoice.load(voice_path, use_cuda=use_cuda)
        self.sample_rate = self.voice.config.sample_rate
        if self.sample_rate != 22050:
            print(f"Warning: voice is {self.sample_rate} Hz, module expects 22050 Hz")

        self.voice_name = os.path.splitext(os.path.basename(voice_path))[0]
        self.caching = caching
        self.tmp_dir = os.path.expanduser(tmp_dir)
        os.makedirs(self.tmp_dir, exist_ok=True)

    def get_cache_path(self, text):
        # include the voice name so switching voices doesn't return old audio
        key = hashlib.md5(f"{self.voice_name}:{text}".encode("utf-8")).hexdigest()
        return os.path.join(self.tmp_dir, f"{key}.raw")

    def synthesize(self, text):
        """Returns the speech as 22050 Hz int16 mono PCM bytes."""
        cache_path = self.get_cache_path(text)
        if self.caching and os.path.isfile(cache_path):
            with open(cache_path, "rb") as cfile:
                return cfile.read()

        waveform = b"".join(
            chunk.audio_int16_bytes for chunk in self.voice.synthesize(text)
        )

        if self.caching:
            with open(cache_path, "wb") as cfile:
                cfile.write(waveform)

        return waveform


class SpeechBrainTTSModule(retico_core.AbstractModule):
    @staticmethod
    def name():
        return "Speechbrain TTS Module"

    @staticmethod
    def description():
        return "A module that synthesizes speech using SpeechBrain."

    @staticmethod
    def input_ius():
        return [retico_core.text.TextIU]

    @staticmethod
    def output_iu():
        return retico_core.audio.AudioIU

    def __init__(
        self, language="en", dispatch_on_finish=True, frame_duration=0.2, **kwargs
    ):
        super().__init__(**kwargs)

        self.dispatch_on_finish = dispatch_on_finish
        self.language = language
        self.tts = PiperTTS(voice_path=VOICE_PATH)
        self.frame_duration = frame_duration
        self.samplerate = 22050  # samplerate of tts (fixed at 22050 for speechbrain)
        self.samplewidth = 2
        self._tts_thread_active = False
        self._latest_text = ""
        self.latest_input_iu = None
        self.audio_buffer = []
        self.audio_pointer = 0
        self.clear_after_finish = False

    def current_text(self):
        return " ".join(iu.text for iu in self.current_input)

    def process_update(self, update_message):
        if not update_message:
            return None
        final = False
        for iu, ut in update_message:
            if ut == retico_core.UpdateType.ADD:
                self.current_input.append(iu)
                self.latest_input_iu = iu
            elif ut == retico_core.UpdateType.REVOKE:
                self.revoke(iu)
            elif ut == retico_core.UpdateType.COMMIT:
                final = True
        current_text = self.current_text()
        if final or (
            len(current_text) - len(self._latest_text) > 15
            and not self.dispatch_on_finish
        ):
            self._latest_text = current_text
            chunk_size = int(self.samplerate * self.frame_duration)
            chunk_size_bytes = chunk_size * self.samplewidth
            new_audio = self.tts.synthesize(current_text)
            new_buffer = []
            i = 0
            while i < len(new_audio):
                chunk = new_audio[i : i + chunk_size_bytes]
                if len(chunk) < chunk_size_bytes:
                    chunk = chunk + b"\x00" * (chunk_size_bytes - len(chunk))
                new_buffer.append(chunk)
                i += chunk_size_bytes
            if self.clear_after_finish:
                self.audio_buffer.extend(new_buffer)
            else:
                self.audio_buffer = new_buffer
        if final:
            self.clear_after_finish = True
            self.current_input = []

    def _tts_thread(self):
        t1 = time.time()
        while self._tts_thread_active:
            t2 = t1
            t1 = time.time()
            if t1 - t2 < self.frame_duration:
                time.sleep(self.frame_duration)
            else:
                time.sleep(max((2 * self.frame_duration) - (t1 - t2), 0))

            if self.audio_pointer >= len(self.audio_buffer):
                raw_audio = (
                    b"\x00"
                    * self.samplewidth
                    * int(self.samplerate * self.frame_duration)
                )
                if self.clear_after_finish:
                    self.audio_pointer = 0
                    self.audio_buffer = []
                    self.clear_after_finish = False
            else:
                raw_audio = self.audio_buffer[self.audio_pointer]
                self.audio_pointer += 1
            iu = self.create_iu(self.latest_input_iu)
            iu.set_audio(raw_audio, 1, self.samplerate, self.samplewidth)
            um = retico_core.UpdateMessage.from_iu(iu, retico_core.UpdateType.ADD)
            self.append(um)

    def prepare_run(self):
        self.audio_pointer = 0
        self.audio_buffer = []
        self._tts_thread_active = True
        self.clear_after_finish = False
        threading.Thread(target=self._tts_thread).start()

    def shutdown(self):
        self._tts_thread_active = False
