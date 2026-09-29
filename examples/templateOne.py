import queue
import time

import numpy as np
import pyaudio
import retico_core
from retico_core.abstract import AbstractModule, AbstractProducingModule, UpdateMessage
from retico_core.audio import AudioIU
from sota_thinclient import ConnectionManager
from sota_thinclient.http_audio_stream import (
    _FIELD_BUFFERSIZE,
    _FIELD_SAMPLERATE,
    _FIELD_SAMPLEWIDTH,
)

SOTA_IP = "10.151.63.79"
HTTP_PORT = "8080"
MIC_UDP_PORT = 52001
OPEN_SECONDS = 5


class PersonalMic(AbstractProducingModule):
    """Reads the computer's microphone (or a headset) in 20 ms chunks."""

    @staticmethod
    def name():
        return "Personal Mic"

    @staticmethod
    def description():
        return "Produces audio from a local microphone"

    @staticmethod
    def output_iu():
        return AudioIU

    def __init__(
        self, rate=16000, chunk_ms=20, device_index=None, device_name=None, **kwargs
    ):
        super().__init__(**kwargs)
        self.rate = rate
        self.chunk = int(rate * chunk_ms / 1000)  # 320 samples = 20 ms
        self.device_index = device_index  # None = system default mic
        self.device_name = device_name  # e.g. "CMF Buds" (part of the name is enough)
        self.pa = None
        self.stream = None

    def setup(self):  # open the mic once, before the thread starts
        self.pa = pyaudio.PyAudio()
        if self.device_name is not None:  # find the input device by name
            for i in range(self.pa.get_device_count()):
                info = self.pa.get_device_info_by_index(i)
                if self.device_name in info["name"] and info["maxInputChannels"] > 0:
                    self.device_index = i
                    break
            else:
                raise ValueError(f"No input device named '{self.device_name}'")
        print(
            f"Personal mic: {self.pa.get_device_info_by_index(self.device_index or self.pa.get_default_input_device_info()['index'])['name']}"
        )
        self.stream = self.pa.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self.rate,
            input=True,
            frames_per_buffer=self.chunk,
            input_device_index=self.device_index,
        )

    def process_update(self, _):  # called in a loop; read() waits for 20 ms of audio
        data = self.stream.read(self.chunk, exception_on_overflow=False)
        iu = self.create_iu()
        iu.set_audio(data, self.chunk, self.rate, 2)
        iu.meta_data["type"] = "user"
        return UpdateMessage.from_iu(iu, retico_core.UpdateType.ADD)

    def shutdown(self):  # close the mic when the module stops
        self.stream.stop_stream()
        self.stream.close()
        self.pa.terminate()


class SotaMicrophoneModule(AbstractProducingModule):
    """Audio from the Sota robot's microphone, streamed over UDP."""

    @staticmethod
    def name():
        return "Sota Microphone Module"

    @staticmethod
    def description():
        return "A producing module that provides audio from a Sota robot, streamed over udp."

    @staticmethod
    def output_iu():
        return AudioIU

    def __init__(
        self, sota: ConnectionManager, data_udp_port: int, buffer_ms: int = 20, **kwargs
    ):
        super().__init__(**kwargs)
        self._sota = sota
        self._rate = None
        self._sample_width = None
        self._data_udp_port = data_udp_port
        self._audio_buffer = sota.microphone.data_queue
        self._frames_per_buffer = None
        self._buffer_ms = buffer_ms
        if buffer_ms % 10 != 0:
            print("Error: use a multiple of 10ms to play nicely with other libraries")

    def process_update(self, _):
        if not self._audio_buffer:
            return None
        try:
            sample = self._audio_buffer.get(timeout=1.0)
        except queue.Empty:
            return None
        iu = self.create_iu()
        iu.set_audio(sample, self._frames_per_buffer, self._rate, self._sample_width)
        iu.meta_data["type"] = "robot"
        return UpdateMessage.from_iu(iu, retico_core.UpdateType.ADD)

    def setup(self):
        self._sota.microphone.enable(
            data_udp_port=self._data_udp_port, restart_if_enabled=True
        )
        sota_state = self._sota.microphone.get_state(use_cached=True)
        print("Initial mic stream state" + str(sota_state))

        self._rate = sota_state[_FIELD_SAMPLERATE]
        self._sample_width = sota_state[_FIELD_SAMPLEWIDTH] // 8

        buffer_size_needed = int(
            self._buffer_ms * self._rate / 1000 * self._sample_width
        )
        if (
            buffer_size_needed != sota_state[_FIELD_BUFFERSIZE]
        ):  # restart with the right buffer
            self._sota.microphone.enable(
                data_udp_port=self._data_udp_port,
                request_buffer_size=buffer_size_needed,
                restart_if_enabled=True,
            )
            sota_state = self._sota.microphone.get_state(use_cached=True)
            print("Updated mic stream state" + str(sota_state))

        self._frames_per_buffer = sota_state[_FIELD_BUFFERSIZE] // self._sample_width

    def prepare_run(self):
        pass

    def shutdown(self):
        self._sota.microphone.disable()


class AudioAnnotator(AbstractModule):
    """Writes rms and seconds into each chunk's meta_data."""

    @staticmethod
    def name():
        return "Audio Annotator"

    @staticmethod
    def description():
        return "Tags each audio chunk with rms and seconds"

    @staticmethod
    def input_ius():
        return [AudioIU]

    @staticmethod
    def output_iu():
        return AudioIU

    def process_update(self, update_msg):
        for iu, ut in update_msg:
            x = np.frombuffer(iu.raw_audio, dtype=np.int16) / 32768
            iu.meta_data["rms"] = float(np.sqrt(np.mean(x**2))) if len(x) else 0.0
            iu.meta_data["seconds"] = len(iu.raw_audio) / (iu.sample_width * iu.rate)
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
        self.keep = keep  # "user" or "robot"

    def process_update(self, update_message):
        out = UpdateMessage()
        for iu, ut in update_message:
            if (
                ut == retico_core.UpdateType.ADD
                and iu.meta_data.get("type") == self.keep
            ):
                out.add_iu(iu, ut)  # keep this type, drop the other
        return out if len(out) else None


import threading
import wave

TARGET_SAMPLES = OPEN_SECONDS * 16000  # exactly OPEN_SECONDS of 16 kHz audio

counts = {"user": 0, "robot": 0}
written = 0  # samples saved so far
done = threading.Event()  # set when the file is full

wav_out = wave.open("kept_audio2.wav", "wb")
wav_out.setnchannels(1)
wav_out.setsampwidth(2)  # int16
wav_out.setframerate(16000)  # both mics are 16 kHz


def count(update_message):
    global written
    for iu, ut in update_message:
        if done.is_set():
            return
        counts[iu.meta_data["type"]] += 1
        samples = len(iu.raw_audio) // 2  # 2 bytes per sample
        take = min(samples, TARGET_SAMPLES - written)  # don't go past the target
        wav_out.writeframes(iu.raw_audio[: take * 2])
        written += take
        if written >= TARGET_SAMPLES:
            done.set()


if __name__ == "__main__":
    sota = ConnectionManager(SOTA_IP, HTTP_PORT)

    pers_mic = PersonalMic()
    sota_mic = SotaMicrophoneModule(sota, MIC_UDP_PORT, buffer_ms=20)
    audio_anno = AudioAnnotator(sleep_interval=0.001)
    user_only = AudioClassifier(keep="user", sleep_interval=0.001)
    user_out = retico_core.debug.CallbackModule(callback=count, sleep_interval=0.001)

    sota_mic.subscribe(audio_anno)  # robot channel in
    pers_mic.subscribe(audio_anno)  # user channel in
    audio_anno.subscribe(user_only)  # keeps user, drops robot
    user_only.subscribe(user_out)  # registers what's left

    modules = [user_out, user_only, audio_anno, pers_mic, sota_mic]
    for m in modules:  # receivers first, mics last
        m.run()

    print(f"recording {OPEN_SECONDS} seconds of audio...")
    start = time.time()
    finished = done.wait(timeout=OPEN_SECONDS * 3)  # wait until the file is full
    took = time.time() - start

    for m in reversed(modules):  # mics first
        m.stop()
    wav_out.close()

    print(f"saved kept_audio.wav: {written / 16000:.2f} s of audio in {took:.2f} s")
    if not finished:
        print("stopped by timeout: the mic delivered less audio than expected")
    print(f"user chunks: {counts['user']}   robot chunks: {counts['robot']}")
