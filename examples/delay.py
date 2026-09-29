import time

import retico_core
from dialogue_control import LoopRecorder
from retico_core.text import TextIU
from retico_speechbraintts import SpeechBrainTTSModule
from sota_thinclient import ConnectionManager

from sota_retico import SotaMicrophoneModule, SotaSpeakerModule

SOTA_IP = "10.151.63.79"
HTTP_PORT = "8080"
MIC_UDP_PORT = 52001
SPEAKER_UDP_PORT = 52002

sota = ConnectionManager(SOTA_IP, HTTP_PORT)

microphone_module = SotaMicrophoneModule(sota, MIC_UDP_PORT, buffer_ms=20)
speaker_module = SotaSpeakerModule(sota, SPEAKER_UDP_PORT)
tts_module = SpeechBrainTTSModule(language="en")
loop = LoopRecorder(tts_module)

# no ASR → no loop: the robot speaks only once
tts_module.subscribe(speaker_module)
tts_module.subscribe(loop)  # y: what the robot says
microphone_module.subscribe(loop)  # x: what the mic hears

for m in [speaker_module, loop, tts_module, microphone_module]:
    m.run()

time.sleep(2)  # let everything start
print("speaking... stay silent")

# inject one sentence into the TTS, as if the ASR had committed it
iu = TextIU(creator=loop, iuid="test:0")
iu.payload = (
    "Hello, this is a short test of the delay between my speaker and my microphone."
)
msg = retico_core.UpdateMessage()
msg.add_iu(iu, retico_core.UpdateType.ADD)
msg.add_iu(iu, retico_core.UpdateType.COMMIT)
tts_module.process_update(msg)

time.sleep(10)  # synthesis + speech + echo

print(f"loop delay (correlation): {loop.estimate_delay():.2f} s")
loop.onset_offset()

for m in [microphone_module, tts_module, loop, speaker_module]:
    m.stop()
