import retico_core
from audio_filter import MarkDecoder, MarkEncoder  # NEW
from audio_manager import WavFile
from execution_manager import ModuleGraph
from retico_huggingfacelm.huggingface_lm_client import HuggingfaceLMClient
from retico_huggingfacelm.huggingface_lm_module import HuggingfaceLMModule
from retico_speechbraintts import SpeechBrainTTSModule
from retico_whisperasr import WhisperASRModule
from text_manager import text_candidate_callback

from sota_retico import SotaMicrophoneModule, SotaSpeakerModule
from sota_thinclient import ConnectionManager

SOTA_IP = "10.151.63.79"
HTTP_PORT = "8080"
MIC_UDP_PORT = 52001
SPEAKER_UDP_PORT = 52002

sota = ConnectionManager(SOTA_IP, HTTP_PORT)

#################initialize the retico modules
sota_mic = SotaMicrophoneModule(sota, MIC_UDP_PORT, buffer_ms=20)
sota_speaker = SotaSpeakerModule(sota, SPEAKER_UDP_PORT)
text_callback_module = retico_core.debug.CallbackModule(
    callback=text_candidate_callback
)
encoder = MarkEncoder()  # NEW: marks the robot's voice
decoder = MarkDecoder(
    threshold=0.5, hold=0.8
)  # NEW: silences marked audio before the asr

print("Starting Whisper ASR...", end="")
asr_module = WhisperASRModule()
print("done.")

print("Starting SpeechBrains...", end="")
tts_module = SpeechBrainTTSModule(language="en")
print("done.")

print("Starting hugging face lm...", end="")
llm_client = HuggingfaceLMClient.quick_from_checkpoint(
    "HuggingFaceTB/SmolLM2-1.7B-Instruct", device="cpu", temperature=0.7
)
llm_module = HuggingfaceLMModule(llm_client)
llm_client.set_system_role("You are a grumpy Wizard who does not talk a lot.")
print("done.")

asr_wav = WavFile("asr_input.wav")  # what the asr receives, to listen to

#################setup the connections
# listen: mic → decoder → (wav file, asr → llm + text callback)
sota_mic.subscribe(decoder)
decoder.subscribe(asr_wav)
decoder.subscribe(asr_module)
asr_module.subscribe(llm_module)
asr_module.subscribe(text_callback_module)
llm_module.subscribe(tts_module)

# speak: tts → encoder → speaker
tts_module.subscribe(encoder)
encoder.subscribe(sota_speaker)

##################start everyone (receivers first, mic last)
modules = [
    sota_speaker,
    encoder,
    tts_module,
    text_callback_module,
    llm_module,
    asr_module,
    asr_wav,
    decoder,
    sota_mic,
]
for m in modules:
    m.run()

input("running... press Enter to stop\n")

for m in reversed(modules):  # mic first, so nothing new enters
    m.stop(clear_buffer=False)  # don't empty queues: avoids the stop() hang
print("stopped")
