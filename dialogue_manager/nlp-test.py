from execution_manager import ModuleGraph
from nlp import PhonemeSubtractor, TextTokenizer
from retico_core.audio import *
from retico_speechbraintts import SpeechBrainTTSModule
from retico_whisperasr import WhisperASRModule

from sota_retico import SotaMicrophoneModule, SotaSpeakerModule
from sota_thinclient import ConnectionManager

# SOTA_IP = "192.168.0.23"
SOTA_IP = "10.151.63.79"
HTTP_PORT = "8080"
MIC_UDP_PORT = 52001
SPEAKER_UDP_PORT = 52002

sota = ConnectionManager(SOTA_IP, HTTP_PORT)

#################initialize the retico modules
microphone_module = SotaMicrophoneModule(sota, MIC_UDP_PORT, buffer_ms=20)
speaker_module = SotaSpeakerModule(sota, SPEAKER_UDP_PORT)

# print("Starting Whisper ASR...", end="")
# asr_module = WhisperASRModule()
# print("done.")

# print("Starting tokenizer and subtractor...", end="")
tok_module = TextTokenizer()
# sub_module = PhonemeSubtractor()
# sub_module.set_known("hello")  # what person A said
# print("done.")

# #################setup the connections
# execution_order = {
#     "sota": {microphone_module: [asr_module, {tok_module: None, sub_module: None}]},
# }


from nlp import PhonemeASRModule, PhonemeSubtractor

phon_asr = PhonemeASRModule()
sub_module = PhonemeSubtractor(input_is_phonemes=True)
sub_module.set_known("Join the conversation to interact")

execution_order = {
    "sota": {microphone_module: [phon_asr, sub_module]},
}


graph = ModuleGraph(execution_order)
graph.show()
graph.run()

input("running... press Enter to stop\n")
graph.stop()  # mic first, so nothing new enters
print("Tokens:  ", tok_module.tokens)
print("Known:   ", sub_module.known)
print("Heard:   ", sub_module.heard)
print("Residual:", sub_module.residual)
