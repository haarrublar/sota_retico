import Levenshtein
from retico_core import AbstractModule, UpdateMessage, UpdateType
from retico_core.audio import AudioIU
from retico_core.text import SpeechRecognitionIU
from retico_core.text import TextIU

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import numpy as np

# Determines speaking state based on the provided speech generation module. When there is non trivial energy
# in the speaker output, suppresses all AudioIUs from propagating. Otherwise they just pass through.
class AudioGatingModule(AbstractModule):

    ENERGY_KEY = "energy"
    START_TALKING_THRESH = 20  # from our energy calculation

    def __init__(
            self,
            speech_gen_module,   ## we use this module's output to determine speaking state.
            **kwargs):
        super().__init__(**kwargs)
        self._is_talking = None
        self._speech_gen_module = speech_gen_module

    @staticmethod
    def name():
        return "Simple Audio Gating Module"

    @staticmethod
    def description():
        return "A simple filter that suppresses IUs from the microphone while the speaker has output."

    @staticmethod
    def input_ius():
        return [AudioIU]

    @staticmethod
    def output_iu():
        return AudioIU

    def process_update(self, update_message):
        output_update = UpdateMessage()
        has_ius = False

        for iu, ut in update_message:

            if iu.creator == self._speech_gen_module:  # use to establish talking state
                self._update_is_talking_check( AudioGatingModule._rms_energy(iu.raw_audio, iu.sample_width) )

            else:  # all other AudioIUs
                if not self._is_talking:
                    output_iu = self.create_iu(iu)
                    output_iu.set_audio(iu.raw_audio, iu.nframes, iu.rate, iu.sample_width)
                    output_update.add_iu(output_iu, ut)
                    has_ius = True

        if has_ius:
            self.append(output_update)

    # very simple threshold that triggers a change when we cross the threshold
    def _update_is_talking_check(self, energy):
        talking = energy > self.START_TALKING_THRESH

        if talking:
            if self._is_talking is None or not self._is_talking:
                print("is speaking changed to: true")
                self._is_talking = True

        else:  #not talking
            if self._is_talking is None or self._is_talking:
                print("is speaking changed to: false")
                self._is_talking = False

    # calculates energy of waveform. primitive
    @staticmethod
    def _rms_energy(audio_bytes, sample_width_bytes):
        sample_width_bits = sample_width_bytes * 8
        samples = np.frombuffer(audio_bytes, dtype=np.dtype(f"int{sample_width_bits}"))
        return np.sqrt(np.mean(samples.astype(np.float64) ** 2))