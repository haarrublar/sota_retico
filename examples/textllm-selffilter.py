from sota_thinclient import ConnectionManager
from retico_huggingfacelm.huggingface_lm import HuggingfaceLM
from examples.debug_utils import text_candidate_callback
from examples.filter import AudioGatingModule
from sota_retico import SotaMicrophoneModule
from sota_retico import SotaSpeakerModule
from retico_whisperasr import WhisperASRModule
from retico_speechbraintts import  SpeechBrainTTSModule
import retico_core

# SOTA_IP = "192.168.0.23"
SOTA_IP = "10.151.63.71"
HTTP_PORT = "8080"
MIC_UDP_PORT = 52001
SPEAKER_UDP_PORT = 52002

sota = ConnectionManager(SOTA_IP, HTTP_PORT)

#################initialize the retico modules
microphone_module = SotaMicrophoneModule(sota, MIC_UDP_PORT, buffer_ms=20)
speaker_module = SotaSpeakerModule(sota, SPEAKER_UDP_PORT)
text_callback_module = retico_core.debug.CallbackModule(callback=text_candidate_callback)

print("Starting Whisper ASR...", end="")
asr_module = WhisperASRModule()
print("done.")

print("Starting SpeechBrains...", end="")
tts_module = SpeechBrainTTSModule(language="en")
print("done.")

print("Starting hugging face lm...",end="")
device = "cuda:0"
# device = "cpu"
llm_module = HuggingfaceLM.from_checkpoint("HuggingFaceTB/SmolLM2-135M-Instruct", device=device)
llm_module.set_system_role("You are a grumpy Wizard. Keep responses concise. Limit responses to 2-3 sentences unless the user explicitly asks for more detail.")
print("Done")

self_filter_module = AudioGatingModule(tts_module, sleep_interval=0.001)

#################setup the connections
microphone_module.subscribe(self_filter_module)
self_filter_module.subscribe(asr_module)
asr_module.subscribe(llm_module)
asr_module.subscribe(text_callback_module)
llm_module.subscribe(tts_module)
tts_module.subscribe(speaker_module)
tts_module.subscribe(self_filter_module)

##################start everyone
speaker_module.run()
self_filter_module.run()
text_callback_module.run()
tts_module.run()
asr_module.run()
microphone_module.run()
llm_module.run()

print("go")
input()   # wait for user key

#clean up and stop
microphone_module.stop()
asr_module.stop()
tts_module.stop()
speaker_module.stop()
self_filter_module.stop()
text_callback_module.stop()
llm_module.stop()