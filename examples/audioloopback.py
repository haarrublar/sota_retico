from retico_core.audio import *
from sota_thinclient import ConnectionManager
from sota_retico import SotaMicrophoneModule
from sota_retico.sota_audio import SotaSpeakerModule

import retico_core

# SOTA_IP = "192.168.0.23"#
SOTA_IP = "10.151.63.71"
HTTP_PORT = "8080"
MIC_UDP_PORT = 52001
SPEAKER_UDP_PORT = 52002

sota = ConnectionManager(SOTA_IP, HTTP_PORT)

#initialize the retico modules
microphone_module = SotaMicrophoneModule(sota, MIC_UDP_PORT, buffer_ms=20)
speaker_module = SotaSpeakerModule(sota, SPEAKER_UDP_PORT)

#setup the connections
microphone_module.subscribe(speaker_module)

#start everyone
speaker_module.run()
microphone_module.run()

print("go")
input()   # wait for user key

#clean up and stop
microphone_module.stop()
speaker_module.stop()
