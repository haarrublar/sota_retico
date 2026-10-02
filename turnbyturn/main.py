import retico_core
from debug_utils import make_text_callback
from execution_manager import ModuleGraph
from retico_whisperasr import WhisperASRModule
from sota_actions import SotaActions
from sota_audio import SotaMicrophoneModule, SotaSpeakerModule
from speaker_gate_wave import SpeakerGatingModule
from speechbrainttsPIPER import SpeechBrainTTSModule

from sota_thinclient import ConnectionManager

SOTA_IP = "10.151.63.79"
HTTP_PORT = "8080"
MIC_UDP_PORT = 52001
SPEAKER_UDP_PORT = 52002

# sota connections
sota = ConnectionManager(SOTA_IP, HTTP_PORT)
sota_mic = SotaMicrophoneModule(sota, MIC_UDP_PORT, buffer_ms=20)

# modules registration
asr_module = WhisperASRModule(
    language="en", vad_aggresiveness=2, silence_threshold=0.85
)
speaker_module = SotaSpeakerModule(sota, SPEAKER_UDP_PORT)
tts_module = SpeechBrainTTSModule(language="en")
voice_gate = SpeakerGatingModule(speaker_module=speaker_module)

# sota actions and aux func
actions = SotaActions(sota, speaker_module=speaker_module, off_delay=1.0)
text_callback_module = retico_core.debug.CallbackModule(
    callback=make_text_callback(show=("live", "commit"), on_commit=actions.mouth_on)
)


connected_nodes = {
    "test-tts": {sota_mic: [voice_gate, asr_module, tts_module, speaker_module]},
    "debug": {asr_module: [text_callback_module]},
}

graph = ModuleGraph(connected_nodes)
graph.show()

graph.run()
input("running... press Enter to stop\n")
graph.stop()
actions.shutdown()
print("stopped")
