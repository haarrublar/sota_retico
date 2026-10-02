import time

import retico_core
from audio_filter import MarkEncoder
from audio_manager import WavFile
from execution_manager import ModuleGraph
from retico_core.text import TextIU
from retico_speechbraintts import SpeechBrainTTSModule

from sota_retico import SotaMicrophoneModule, SotaSpeakerModule
from sota_thinclient import ConnectionManager

SOTA_IP = "10.151.63.79"
HTTP_PORT = "8080"
MIC_UDP_PORT = 52001
SPEAKER_UDP_PORT = 52002

sota = ConnectionManager(SOTA_IP, HTTP_PORT)
sota_mic = SotaMicrophoneModule(sota, MIC_UDP_PORT, buffer_ms=20)
sota_speaker = SotaSpeakerModule(sota, SPEAKER_UDP_PORT)
tts_module = SpeechBrainTTSModule(language="en")
encoder = MarkEncoder()
air_wav = WavFile("air.wav", sleep_interval=0.001)
tts_wav = WavFile("tts_marked.wav", sleep_interval=0.001)

execution_order = {
    "listen": {sota_mic: air_wav},
    "speak": {tts_module: [encoder, {sota_speaker: None, tts_wav: None}]},
}
graph = ModuleGraph(execution_order)
graph.run()

time.sleep(3)  # 3 s of room silence first (reference)
iu = TextIU(creator=encoder, iuid="say:0")
iu.payload = (
    "Hello, I am Sota. This sentence is a test of my marked voice. "
    "Please stay quiet while I speak, so the microphone only hears me."
)
msg = retico_core.UpdateMessage()
msg.add_iu(iu, retico_core.UpdateType.ADD)
msg.add_iu(iu, retico_core.UpdateType.COMMIT)
tts_module.process_update(msg)  # make Sota speak once

time.sleep(15)  # synthesis + speech + a few seconds of silence after
graph.stop()
print("done")
