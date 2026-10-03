import retico_core
from debug_utils import make_text_callback
from execution_manager import ModuleGraph
from gate import AudioGatingModule
from retico_whisperasr import WhisperASRModule
from sota_actions import SotaActions
from sota_audio import SotaMicrophoneModule, SotaSpeakerModule
from speechbrainttsPIPER import SpeechBrainTTSModule
from timeline import plot_timeline
from user_microphone import UserMicrophoneModule

from sota_thinclient import ConnectionManager

SOTA_IP = "10.151.63.79"
HTTP_PORT = "8080"
MIC_UDP_PORT = 52001
SPEAKER_UDP_PORT = 52002

# sota
sota = ConnectionManager(SOTA_IP, HTTP_PORT)
sota_mic = SotaMicrophoneModule(sota, MIC_UDP_PORT, buffer_ms=20)
speaker = SotaSpeakerModule(sota, SPEAKER_UDP_PORT)
actions = SotaActions(sota, speaker_module=speaker, off_delay=1.0)

# pipeline
gate = AudioGatingModule(speaker_module=speaker)
asr = WhisperASRModule(language="en", vad_aggresiveness=2, silence_threshold=0.85)
tts = SpeechBrainTTSModule(language="en")


def on_commit(text):
    gate.expect_reply()  # user finished, sota's turn


debug = retico_core.debug.CallbackModule(
    callback=make_text_callback(show=("live", "commit"), on_commit=on_commit)
)

user_mic = UserMicrophoneModule()  # only recorded, not used by the pipeline
pc_sink = retico_core.debug.CallbackModule(callback=lambda um: None)

graph = ModuleGraph(
    {
        "echo": {sota_mic: [gate, asr, tts, speaker]},
        "pc": {user_mic: [pc_sink]},
        "debug": {asr: [debug]},
    }
)
graph.show()

graph.run()
input("running... press Enter to stop\n")
graph.stop()
plot_timeline()
actions.shutdown()
print("stopped")
