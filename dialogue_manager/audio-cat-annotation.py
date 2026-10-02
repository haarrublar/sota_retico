import time

import retico_core
from audio_manager import AudioAnnotator, AudioClassifier, WavFile
from execution_manager import ModuleGraph
from retico_core.text import TextIU
from retico_speechbraintts import SpeechBrainTTSModule
from sota_audio import SotaMicrophoneModule
from user_microphone import UserMicrophoneModule

from sota_retico import SotaSpeakerModule
from sota_thinclient import ConnectionManager

SOTA_IP = "10.151.63.79"
HTTP_PORT = "8080"
MIC_UDP_PORT = 52001
SPEAKER_UDP_PORT = 52002

sota = ConnectionManager(SOTA_IP, HTTP_PORT)

# sota channel
sota_mic = SotaMicrophoneModule(sota, MIC_UDP_PORT, buffer_ms=20)
mic_annotator = AudioAnnotator()
sota_audio = AudioClassifier(keep="sota")
sota_wav = WavFile("sota.wav")


# robot voice
tts_module = SpeechBrainTTSModule(language="en")
sota_speaker = SotaSpeakerModule(sota, SPEAKER_UDP_PORT)

execution_order = {
    "sota": {sota_mic: [mic_annotator, sota_audio, sota_wav]},
    "speak": {tts_module: sota_speaker},  # tts → speaker
}


def say(text):  # make Sota speak, as if the asr had committed `text`
    iu = TextIU(creator=sota_audio, iuid=f"say:{time.time()}")
    iu.payload = text
    msg = retico_core.UpdateMessage()
    msg.add_iu(iu, retico_core.UpdateType.ADD)
    msg.add_iu(iu, retico_core.UpdateType.COMMIT)
    tts_module.process_update(msg)


graph = ModuleGraph(execution_order)
graph.show()
graph.run()  # everything running before Sota speaks

time.sleep(2)  # a few seconds of room silence first
say("Hello, I am Sota. This is a test of my voice for the recording.")

input("running... press Enter to stop\n")
graph.stop()
print("saved sota.wav and user.wav")
