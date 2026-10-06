import retico_core
from debug_utils import make_text_callback
from execution_manager import ModuleGraph
from gate import AudioGatingModule
from huggingface_lm_client import HuggingfaceLMClient
from huggingface_lm_module import HuggingfaceLMModule

# from retico_huggingfacelm.huggingface_lm_client import HuggingfaceLMClient
# from retico_huggingfacelm.huggingface_lm_module import HuggingfaceLMModule
from retico_whisperasr import WhisperASRModule
from sota_actions import SotaActions
from sota_audio import SotaMicrophoneModule, SotaSpeakerModule
from speechbrainttsPIPER import SpeechBrainTTSModule

from sota_thinclient import ConnectionManager

SOTA_IP = "10.151.63.79"
HTTP_PORT = "8080"
MIC_UDP_PORT = 52001
SPEAKER_UDP_PORT = 52002

# LLM settings
LLM_CHECKPOINT = "Qwen/Qwen2.5-0.5B-Instruct"
LLM_DEVICE = "cpu"  # "cuda:0" on a machine with an NVIDIA GPU
LLM_TEMPERATURE = 0.7
SYSTEM_ROLE = "You are Sota, a friendly robot. Answer in one or two short sentences."

# sota
sota = ConnectionManager(SOTA_IP, HTTP_PORT)
sota_mic = SotaMicrophoneModule(sota, MIC_UDP_PORT, buffer_ms=20)
speaker = SotaSpeakerModule(sota, SPEAKER_UDP_PORT)
actions = SotaActions(sota, speaker_module=speaker, off_delay=1.0)

# pipeline
print("Loading LLM...", end="", flush=True)
llm_client = HuggingfaceLMClient.quick_from_checkpoint(
    LLM_CHECKPOINT, device=LLM_DEVICE, temperature=LLM_TEMPERATURE
)
llm_client.set_system_role(SYSTEM_ROLE)
llm = HuggingfaceLMModule(llm_client)
print("done.")

gate = AudioGatingModule(speaker_module=speaker, llm_module=llm)
asr = WhisperASRModule(language="en", vad_aggresiveness=2, silence_threshold=0.85)
tts = SpeechBrainTTSModule(language="en")


def on_commit(text):
    gate.expect_reply()  # user finished, sota's turn


debug = retico_core.debug.CallbackModule(
    callback=make_text_callback(show=("live", "commit"), on_commit=on_commit)
)


graph = ModuleGraph(
    {
        "dialogue": {sota_mic: [gate, asr, llm, tts, speaker]},
        "debug": {asr: [debug]},
    }
)
graph.show()

graph.run()
input("running... press Enter to stop\n")
graph.stop()
actions.shutdown()
print("stopped")
