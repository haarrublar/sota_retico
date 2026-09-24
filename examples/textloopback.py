import debug_utils
import matplotlib.pyplot as plt
import numpy as np
import retico_core
from debug_utils import (
    audio_array_callback,
    text_candidate_callback,
)
from retico_core.audio import *
from retico_speechbraintts import SpeechBrainTTSModule
from retico_whisperasr import WhisperASRModule
from sota_thinclient import ConnectionManager

from sota_retico import SotaMicrophoneModule, SotaSpeakerModule

# SOTA_IP = "192.168.0.23"
SOTA_IP = "10.0.0.12"
# SOTA_IP = "10.151.63.79"
HTTP_PORT = "8080"
MIC_UDP_PORT = 52001
SPEAKER_UDP_PORT = 52002

sota = ConnectionManager(SOTA_IP, HTTP_PORT)

#################initialize the retico modules
microphone_module = SotaMicrophoneModule(sota, MIC_UDP_PORT, buffer_ms=20)

# callbacks
speaker_module = SotaSpeakerModule(sota, SPEAKER_UDP_PORT)
text_callback_module = retico_core.debug.CallbackModule(
    callback=text_candidate_callback
)
wav_array_callback_module = retico_core.debug.CallbackModule(
    callback=audio_array_callback
)

print("Starting Whisper ASR...", end="")
asr_module = WhisperASRModule()
print("done.")

print("Starting SpeechBrains...", end="")
tts_module = SpeechBrainTTSModule(language="en")
print("done.")

#################setup the connections
debug_utils.recording = True
microphone_module.subscribe(asr_module)
microphone_module.subscribe(wav_array_callback_module)
asr_module.subscribe(tts_module)
asr_module.subscribe(text_callback_module)
tts_module.subscribe(speaker_module)

##################start everyone
speaker_module.run()
wav_array_callback_module.run()
text_callback_module.run()
tts_module.run()
asr_module.run()
microphone_module.run()

print("go")
input()  # wait for user key

THRESHOLD = 0.008
for i, seg in enumerate(debug_utils.audio_info):
    if seg["rms"] < THRESHOLD:
        print(seg)

microphone_module.stop()
asr_module.stop()
tts_module.stop()
speaker_module.stop()
text_callback_module.stop()
debug_utils.recording = False
wav_array_callback_module.stop()
