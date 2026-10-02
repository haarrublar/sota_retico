import time

import retico_core
from audio_filter import MarkDecoder, MarkEncoder
from audio_manager import WavFile
from execution_manager import ModuleGraph
from retico_core.text import TextIU
from retico_speechbraintts import SpeechBrainTTSModule

from sota_retico import SotaMicrophoneModule, SotaSpeakerModule
from sota_thinclient import ConnectionManager

sota = ConnectionManager("10.151.63.79", "8080")
sota_mic = SotaMicrophoneModule(sota, 52001, buffer_ms=20)
sota_speaker = SotaSpeakerModule(sota, 52002)
tts_module = SpeechBrainTTSModule(language="en")
encoder = MarkEncoder(width_hz=150)

execution_order = {
    "speak": {tts_module: [encoder, sota_speaker]},  # Sota's voice gets the mark
    "save": {
        sota_mic: {
            WavFile("mic_all.wav"): None,  # everything the mic heard, to compare
            MarkDecoder(mode="keep"): WavFile("sota_voice.wav"),  # only Sota
        }
    },
}

graph = ModuleGraph(execution_order)
graph.show()
graph.run()


def say(text):  # make Sota speak, as if the asr had committed `text`
    iu = TextIU(creator=encoder, iuid=f"say:{time.time()}")
    iu.payload = text
    msg = retico_core.UpdateMessage()
    msg.add_iu(iu, retico_core.UpdateType.ADD)
    msg.add_iu(iu, retico_core.UpdateType.COMMIT)
    tts_module.process_update(msg)


time.sleep(2)  # a few seconds of room silence first
say("Hello, I am Sota. This is a test of my voice for the recording.")

input("talk too if you want, then press Enter to stop\n")
graph.stop()
print("saved mic_all.wav and sota_voice.wav")
