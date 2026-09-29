import pyaudio
import retico_core
from retico_core.abstract import AbstractProducingModule, UpdateMessage
from retico_core.audio import AudioIU


class UserMicrophoneModule(AbstractProducingModule):
    """Reads the user's microphone (or a headset) in 20 ms chunks."""

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
        self.device_name = device_name
        self.pa = None
        self.stream = None

    def setup(self):  # open the mic once, before the thread starts
        self.pa = pyaudio.PyAudio()
        if self.device_name is not None:
            for i in range(self.pa.get_device_count()):
                info = self.pa.get_device_info_by_index(i)
                name = str(info["name"])
                max_channels = int(info["maxInputChannels"])
                if self.device_name in name and max_channels >= 0:
                    self.device_index = i
                    break
            else:
                raise ValueError(f"No input device named '{self.device_name}'")
        if self.device_index is None:
            self.device_index = int(self.pa.get_default_input_device_info()["index"])
        mic_name = self.pa.get_device_info_by_index(self.device_index)["name"]
        print(f"Personal mic: {mic_name}")
        self.stream = self.pa.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self.rate,
            input=True,
            frames_per_buffer=self.chunk,
            input_device_index=self.device_index,
        )

    def process_update(self, _):
        if self.stream is not None:
            data = self.stream.read(self.chunk, exception_on_overflow=False)
        iu = self.create_iu()
        iu.set_audio(data, self.chunk, self.rate, 2)
        iu.meta_data["owner"] = "user"
        return UpdateMessage.from_iu(iu, retico_core.UpdateType.ADD)

    def prepare_run(self):
        pass

    def shutdown(self):
        if self.stream is not None:
            self.stream.stop_stream()
            self.stream.close()
        if self.pa is not None:
            self.pa.terminate()
