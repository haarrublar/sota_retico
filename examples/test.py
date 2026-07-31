from retico_core.audio import *
from retico_whisperasr import WhisperASRModule
from retico_speechbraintts import  SpeechBrainTTSModule
from retico_huggingfacelm.huggingface_lm import HuggingfaceLM
from sota_thinclient import ConnectionManager
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer

from sota_retico import SotaMicrophoneModule

import retico_core

from sota_retico.sota_audio import SotaSpeakerModule

print("Starting hugging face lm...",end="")
device = "cuda:0"
# device = "cpu"
""" HuggingFace Model, Tokenzier, Model """
checkpoint = "HuggingFaceTB/SmolLM2-135M-Instruct"
tokenizer = AutoTokenizer.from_pretrained(checkpoint,  trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(checkpoint,  trust_remote_code=True).to(device)
streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
print("Done")

# SOTA_IP = "192.168.0.23"#
SOTA_IP = "10.151.63.71"
HTTP_PORT = "8080"
MIC_UDP_PORT = 52001
SPEAKER_UDP_PORT = 52002


msg = []
def callback(update_msg):
    global msg
    print("update: ", end="")
    for x, ut in update_msg:
        if ut == retico_core.UpdateType.ADD:
            msg.append(x)
            print(x.text+" ", end="")
        if ut == retico_core.UpdateType.REVOKE:
            msg.remove(x)
    print("")

    txt = "Committed: "
    committed = False
    for x in msg:
        txt += x.text + " "
        committed = committed or x.committed
    print(" " * 80, end="\r")
    print(f"{txt}", end="\r")
    if committed:
        msg = []
        print("")

sota = ConnectionManager(SOTA_IP, HTTP_PORT)
# microphone_module = MicrophoneModule(rate=16000, sample_width=2)
microphone_module = SotaMicrophoneModule(sota, MIC_UDP_PORT, buffer_ms=20)
# speaker_module = SpeakerModule(rate=22050, sample_width=2)
speaker_module = SotaSpeakerModule(sota, SPEAKER_UDP_PORT)

print("Starting Whisper ASR...", end="")
asr = WhisperASRModule()
print("done.")

m3 = retico_core.debug.CallbackModule(callback=callback)
# m4 = retico_core.debug.TextPrinterModule()

print("Starting SpeechBrains...", end="")
tts = SpeechBrainTTSModule(language="en")
print("done.")

lm = HuggingfaceLM("cuda:0", tokenizer, model, streamer)

# microphone_module.subscribe(speaker_module)
microphone_module.subscribe(asr)
asr.subscribe(lm)
# lm.subscribe(m4)
lm.subscribe(tts)
asr.subscribe(m3)
# asr.subscribe(tts)
tts.subscribe(speaker_module)

# asr.subscribe(m4)

microphone_module.run()
asr.run()
tts.run()
lm.run()
speaker_module.run()
m3.run()
# m4.run()
print("go")
input()

microphone_module.stop()
speaker_module.stop()
asr.stop()
tts.stop()
lm.stop()