import retico_core
from debug_utils import text_candidate_callback
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
text_callback_module = retico_core.debug.CallbackModule(
    callback=text_candidate_callback
)

print("Starting Whisper ASR...", end="")
asr_module = WhisperASRModule()
print("done.")

print("Starting SpeechBrains...", end="")
tts_module = SpeechBrainTTSModule(language="en")
print("done.")

#################setup the connections
microphone_module.subscribe(asr_module)
asr_module.subscribe(tts_module)
asr_module.subscribe(text_callback_module)
tts_module.subscribe(speaker_module)

##################start everyone
speaker_module.run()
text_callback_module.run()
tts_module.run()
asr_module.run()
microphone_module.run()

print("go")
input()  # wait for user key

# clean up and stop
microphone_module.stop()
asr_module.stop()
tts_module.stop()
speaker_module.stop()
text_callback_module.stop()

# import retico_core
# from debug_utils import text_candidate_callback
# from dialogue_control import MicFilter, RobotTag
# from retico_core.audio import *
# from retico_speechbraintts import SpeechBrainTTSModule
# from retico_whisperasr import WhisperASRModule
# from sota_thinclient import ConnectionManager

# from sota_retico import SotaMicrophoneModule, SotaSpeakerModule

# # SOTA_IP = "192.168.0.23"
# SOTA_IP = "10.151.63.79"
# HTTP_PORT = "8080"
# MIC_UDP_PORT = 52001
# SPEAKER_UDP_PORT = 52002

# sota = ConnectionManager(SOTA_IP, HTTP_PORT)

# #################initialize the retico modules
# microphone_module = SotaMicrophoneModule(sota, MIC_UDP_PORT, buffer_ms=20)
# speaker_module = SotaSpeakerModule(sota, SPEAKER_UDP_PORT)
# text_callback_module = retico_core.debug.CallbackModule(
#     callback=text_candidate_callback
# )

# print("Starting Whisper ASR...", end="")
# asr_module = WhisperASRModule()
# print("done.")

# print("Starting SpeechBrains...", end="")
# tts_module = SpeechBrainTTSModule(language="en")
# print("done.")

# robot_tag = RobotTag()  # tags speaker audio as robot
# mic_filter = MicFilter()  # deletes mic audio while the robot is sounding

# #################setup the connections
# microphone_module.subscribe(mic_filter)  # mic → filter
# mic_filter.subscribe(asr_module)  # filter → asr (user only)
# asr_module.subscribe(tts_module)
# asr_module.subscribe(text_callback_module)
# tts_module.subscribe(robot_tag)  # tts → robot tag
# robot_tag.subscribe(speaker_module)  # robot tag → speaker
# robot_tag.subscribe(mic_filter)  # robot tag → mic filter

# ##################start everyone (receivers first, mic last)
# speaker_module.run()
# text_callback_module.run()
# mic_filter.run()
# robot_tag.run()
# tts_module.run()
# asr_module.run()
# microphone_module.run()

# print("go:  talk to Sota,  Enter = quit")
# input()

# # clean up and stop (mic first, so nothing new enters)
# microphone_module.stop()
# asr_module.stop()
# tts_module.stop()
# robot_tag.stop()
# mic_filter.stop()
# speaker_module.stop()
# text_callback_module.stop()


# # import retico_core
# # from debug_utils import text_candidate_callback
# # from dialogue_control import (  # CHANGED
# #     AudioGate,
# #     AudioMeter,
# #     RobotTalking,
# #     TurnTagger,
# # )
# # from retico_core.audio import *
# # from retico_speechbraintts import SpeechBrainTTSModule
# # from retico_whisperasr import WhisperASRModule
# # from sota_thinclient import ConnectionManager

# # from sota_retico import SotaMicrophoneModule, SotaSpeakerModule

# # # SOTA_IP = "192.168.0.23"
# # SOTA_IP = "10.151.63.79"
# # HTTP_PORT = "8080"
# # MIC_UDP_PORT = 52001
# # SPEAKER_UDP_PORT = 52002

# # sota = ConnectionManager(SOTA_IP, HTTP_PORT)

# # #################initialize the retico modules
# # microphone_module = SotaMicrophoneModule(sota, MIC_UDP_PORT, buffer_ms=20)
# # speaker_module = SotaSpeakerModule(sota, SPEAKER_UDP_PORT)
# # text_callback_module = retico_core.debug.CallbackModule(
# #     callback=text_candidate_callback
# # )

# # print("Starting Whisper ASR...", end="")
# # asr_module = WhisperASRModule()
# # print("done.")

# # print("Starting SpeechBrains...", end="")
# # tts_module = SpeechBrainTTSModule(language="en")
# # print("done.")

# # tagger = TurnTagger("robot_talking")  # reads states, writes the tag
# # mic_gate = AudioGate()  # reads the tag, mutes robot chunks
# # tts_meter = AudioMeter()  # NEW: writes rms on each TTS chunk
# # robot_talking = RobotTalking("robot_talking", hangover=2.0)


# # #################setup the connections
# # microphone_module.subscribe(tagger)  # mic → tagger
# # tagger.subscribe(mic_gate)  # tagger → gate
# # mic_gate.subscribe(asr_module)  # gate → asr
# # asr_module.subscribe(tts_module)
# # asr_module.subscribe(text_callback_module)
# # asr_module.subscribe(robot_talking)  # asr text → start of the robot turn
# # tts_module.subscribe(speaker_module)
# # tts_module.subscribe(tts_meter)  # tts → meter
# # tts_meter.subscribe(robot_talking)  # meter → end of the robot turn

# # ##################start everyone (receivers first, mic last)
# # speaker_module.run()
# # robot_talking.run()  # NEW
# # tts_meter.run()  # NEW
# # text_callback_module.run()
# # tts_module.run()
# # asr_module.run()
# # mic_gate.run()
# # tagger.run()
# # microphone_module.run()

# # print("go:  talk to Sota,  Enter = quit")  # CHANGED: no manual switch anymore
# # input()

# # # clean up and stop (mic first, so nothing new enters)
# # microphone_module.stop()
# # tagger.stop()
# # mic_gate.stop()
# # asr_module.stop()
# # tts_module.stop()
# # tts_meter.stop()  # NEW
# # robot_talking.stop()  # NEW
# # speaker_module.stop()
# # text_callback_module.stop()
