"""
Sota Audio Module
============
-- modified from the retico standard audio.py module

This module defines basic incremental units and incremental modules to handle
audio input and output via the Sota
"""

import queue
import threading

import numpy as np
import pyaudio
import retico_core
import time

from retico_core import AbstractModule, AbstractProducingModule, AbstractConsumingModule, UpdateMessage, \
    AbstractTriggerModule, IncrementalUnit, UpdateType
from retico_core.audio import AudioIU
import sota_thinclient
from sota_thinclient import ConnectionManager
from sota_thinclient.http_audio_stream import _FIELD_SAMPLERATE, _FIELD_SAMPLEWIDTH, _FIELD_BUFFERSIZE, \
    StreamingMonoResampler

CHANNELS = 1
"""Number of channels. For now, this is hard coded MONO. If there is interest to do
stereo or audio with even more channels, it has to be integrated into the modules."""

class SotaMicrophoneModule(AbstractProducingModule):

    """A module that produces IUs containing audio signals incoming from a Sota via the sota_thinclient module,
       streamed over a network."""

    @staticmethod
    def name():
        return "Sota Microphone Module"

    @staticmethod
    def description():
        return "A producing module that provides audio from a Sota robot, streamed over udp."

    @staticmethod
    def output_iu():
        return AudioIU

    @staticmethod     # the microphone auto-mutes when other AudioUI is active.
    def input_ius():
        return [AudioIU]

    def callback(self, in_data, frame_count, time_info, status):
        """The callback function that gets called by pyaudio.

        Args:
            in_data (bytes[]): The raw audio that is coming in from the
                microphone
            frame_count (int): The number of frames that are stored in in_data
            data_udp_port (int): The UDP port to listen on and to ask SOTA to send to
            buffer_ms (int): how many ms for the buffer. An underlying library requires a multiple of 10ms
                                <20ms seems to choke with no cpu load, unclear why
        """
        self._audio_buffer.put(in_data)
        return (in_data, pyaudio.paContinue)

    def __init__(self, sota: ConnectionManager,
                 data_udp_port: int,
                 buffer_ms : int = 20 , **kwargs):
        """
        Initialize the Sota Microphone Module.

        Args:
            frame_length (float): The length of one frame (i.e., IU) in seconds
            rate (int): The frame rate of the recording
            sample_width (int): The width of a single sample of audio in bytes.
            device_index (int): The device index of the microphone to use. If None,
                uses the default input device.
        """
        super().__init__(**kwargs)
        self._sota = sota

        self._frame_length = None
        self._rate = None
        self._sample_width = None
        self._data_udp_port = data_udp_port

        self._audio_buffer = sota.microphone.data_queue
        self._frames_per_buffer = None
        self._buffer_ms = buffer_ms
        if not (buffer_ms % 10 == 0):
            print ("Error: use a multiple of 10ms to play nicely with other libraries")

    def process_update(self, update_message):

        if not self._audio_buffer:
            return None
        try:
            sample = self._audio_buffer.get(timeout=1.0)
        except queue.Empty:
            return None

        # print("packet")
        output_iu = self.create_iu()
        output_iu.set_audio(sample, self._frames_per_buffer, self._rate, self._sample_width)
        return retico_core.UpdateMessage.from_iu(output_iu, retico_core.UpdateType.ADD)

    def setup(self):
        """Set up the microphone for recording."""
        self._sota.microphone.enable(data_udp_port=self._data_udp_port, restart_if_enabled=True)
        sota_state = self._sota.microphone.get_state(use_cached=True)
        print("Initial mic stream state"+str(sota_state))

        self._rate = sota_state[_FIELD_SAMPLERATE]
        self._sample_width = sota_state[_FIELD_SAMPLEWIDTH] // 8

        buffer_size_needed = int(self._buffer_ms * self._rate / 1000 * self._sample_width)

        if buffer_size_needed != sota_state[_FIELD_BUFFERSIZE]:  # we need to restart with a different buffer size
            self._sota.microphone.enable(data_udp_port=self._data_udp_port,
                                         request_buffer_size=buffer_size_needed,
                                         restart_if_enabled=True)
            sota_state = self._sota.microphone.get_state(use_cached=True)
            print("Updated mic stream state" + str(sota_state))

        self._frames_per_buffer = sota_state[_FIELD_BUFFERSIZE] // self._sample_width

    def prepare_run(self):
        pass

    def shutdown(self):
        self._sota.microphone.disable()

class SotaSpeakerModule(AbstractConsumingModule):
    """A module that consumes AudioIUs of arbitrary size and outputs them to the
    Sota's speaker over the network.
     When a new IU is incoming, the module blocks as   ******** NOPE
    long as the current IU is being played."""

    @staticmethod
    def name():
        return "Sota Speaker Module"

    @staticmethod
    def description():
        return "A consuming module that plays audio from a connected VStone Sota."

    @staticmethod
    def input_ius():
        return [AudioIU]

    @staticmethod
    def output_iu():
        return None

    def __init__(
        self,
        sota: ConnectionManager,
        data_udp_port: int,
        output_sample_rate : int = None,  # what to tell the Sota to use. None defaults to not asking
        output_sample_width : int = None,
        speaking_trigger: AbstractTriggerModule = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self._has_incoming_audio_params = False
        self._incoming_sample_rate = None
        self._incoming_sample_width = None

        self._sota = sota
        self._data_udp_port = data_udp_port
        self._output_sample_rate = output_sample_rate
        self._output_sample_width = output_sample_width

        self._audio_buffer = sota.speaker.data_queue

        self._resampler = None
        self._currently_making_noise = False
        self._starting_output_trigger = speaking_trigger

    def _confirm_input_audio_params(self):
        self._has_incoming_audio_params = True

    def _setup_sampler(self):
        self._resampler =  StreamingMonoResampler(
            source_rate=self._incoming_sample_rate,
            source_dtype=np.dtype(f'int{self._incoming_sample_width}'),
            target_rate=self._output_sample_rate,
            target_dtype=np.dtype(f'int{self._output_sample_width}')
        )

    # last = None  # debug code
    # counter = 0
    def process_update(self, update_message):

        for iu, ut in update_message:
            if not self._has_incoming_audio_params:
                self._incoming_sample_width = iu.sample_width*8  # we are working in bits for resampling
                self._incoming_sample_rate = iu.rate
                self._has_incoming_audio_params = True
                self._setup_sampler()

            if ut == retico_core.UpdateType.ADD:
                # if self.counter==0:  print("message len "+ str(len(iu.raw_audio)))
                resampled = self._resampler.resample_chunk(bytes(iu.raw_audio))
                self._audio_buffer.put(resampled, block=False)

                if self._starting_output_trigger is not None:
                    self._starting_output_trigger.trigger(
                        data = {
                            SpeakerTrigger.ENERGY_KEY: self.rms_energy(resampled, self._incoming_sample_width)
                        }
                    )

        return None

    def rms_energy(self, audio_bytes, sample_width_bits):
        samples = np.frombuffer(audio_bytes, dtype=np.dtype(f"int{sample_width_bits}"))
        return np.sqrt(np.mean(samples.astype(np.float64) ** 2))

    def setup(self):
        self._sota.speaker.enable(data_udp_port=self._data_udp_port)
        state = self._sota.speaker.get_state(use_cached=True)
        self._output_sample_rate = state[_FIELD_SAMPLERATE]
        self._output_sample_width =state[_FIELD_SAMPLEWIDTH]

    def shutdown(self):
        self._sota.speaker.disable()


class SpeakerStateIU(IncrementalUnit):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.is_speaking = False

    @staticmethod
    def type():
        return "speaker_state"


class SpeakerTrigger(AbstractTriggerModule):

    ENERGY_KEY = "energy"
    START_TALKING_THRESH = 20  # from our energy calculation

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._is_talking = None

    @staticmethod
    def output_iu():
        return SpeakerStateIU

    # Call trigger only with is_talking=True, as it will schedule the end False trigger automatically
    #    based on the duration
    def trigger(self, data=None, update_type=UpdateType.ADD):
        if not self.ENERGY_KEY in data: return  # no valid data

        energy = data[self.ENERGY_KEY]
        talking = energy > SpeakerTrigger.START_TALKING_THRESH  # potential stop talking...

        if talking:
            if self._is_talking is None or not self._is_talking:
                print("is speaking changed to: true")
                self._is_talking = True
                iu = self.create_iu()
                iu.is_speaking = True
                self.append(UpdateMessage.from_iu(iu, update_type))

        else:  #not talking
            if self._is_talking is None or self._is_talking:
                print("is speaking changed to: false")
                self._is_talking = False
                iu = self.create_iu()
                iu.is_speaking = False
                self.append(UpdateMessage.from_iu(iu, update_type))


class AudioGatingModule(AbstractModule):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._isSpeaking = False

    @staticmethod
    def name():
        return "Simple Audio Gating Module"

    @staticmethod
    def description():
        return "A simple filter that suppresses IUs from the microphone while the speaker has output."

    @staticmethod
    def input_ius():
        return [AudioIU, SpeakerStateIU]

    @staticmethod
    def output_iu():
        return AudioIU

    def process_update(self, update_message):
        output_update = UpdateMessage()
        has_ius = False

        for iu, ut in update_message:

            if isinstance(iu, AudioIU):
                if not self._isSpeaking:
                    output_iu = self.create_iu(iu)
                    output_iu.set_audio(iu.raw_audio, iu.nframes, iu.rate, iu.sample_width)
                    output_update.add_iu(output_iu, ut)
                    has_ius = True

            elif isinstance(iu, SpeakerStateIU):
                print("got speaker state IU")
                self._isSpeaking = iu.is_speaking

        if has_ius:
            self.append(output_update)
