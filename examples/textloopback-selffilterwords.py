from retico_core import AbstractModule, UpdateMessage, UpdateType
from retico_core.text import SpeechRecognitionIU, TextIU
from sota_thinclient import ConnectionManager

from examples.filter import ASRSimilarityFilterModule
from sota_retico import SotaMicrophoneModule
from sota_retico.sota_audio import SotaSpeakerModule
from retico_whisperasr import WhisperASRModule
from retico_speechbraintts import  SpeechBrainTTSModule
import retico_core
# from filter import RobotASRFilterModule, SimpleTextPassthrough

SOTA_IP = "192.168.0.23"
# SOTA_IP = "10.151.63.71"
HTTP_PORT = "8080"
MIC_UDP_PORT = 52001
SPEAKER_UDP_PORT = 52002


### A debugging callback function that retico can send data to.
# here we use it as a way to see what the ASR is picking up. It's interesting and useful for debugging.
msg = []
def text_candidate_callback(update_msg):
    global msg

    for x, ut in update_msg:    #update_msg is  (IU, IU type)
        if ut == retico_core.UpdateType.ADD:
            msg.append(x)

        if ut == retico_core.UpdateType.REVOKE:
            if x in msg:
                msg.remove(x)

    # calculate the committed message so far
    txt = ""
    committed = False
    for x in msg:
        txt += x.text + " "
        committed = committed or x.committed

    if committed:
        msg = []
        print("\nCommitted: "+txt)

    else:
        print("\rlive: "+txt, end="")

## A dummy module for loopback into the ASR to filter what the robot said last. Only relevant for this dummy loop
# example
class CommittedTextRepeater(AbstractModule):
    def __init__(self):
        super().__init__()

    @staticmethod
    def name():
        return "Passthrough committed text only"

    @staticmethod
    def description():
        return "A dummy module that passes text through committed but changes the type to text, e.g., if its ASR."

    @staticmethod
    def input_ius():
        return [SpeechRecognitionIU]

    @staticmethod
    def output_iu():
        return TextIU

    def process_update(self, update: UpdateMessage):
        output_update = UpdateMessage()

        for input_iu, ut in update:
            if ut != UpdateType.COMMIT:
                continue
            output_iu = self.create_iu(input_iu)

            output_iu.text = input_iu.text
            if hasattr(input_iu, "final"):
                output_iu.final = input_iu.final

            if hasattr(input_iu, "committed"):
                output_iu.committed = input_iu.committed

            output_update.add_iu(output_iu, ut)

        self.append(output_update)


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

loopback_repeater = CommittedTextRepeater()
self_filter_module = ASRSimilarityFilterModule(sleep_interval=0.001)

#################setup the connections
microphone_module.subscribe(asr_module)
asr_module.subscribe(self_filter_module)
self_filter_module.subscribe(tts_module)
tts_module.subscribe(speaker_module)

asr_module.subscribe(loopback_repeater)
loopback_repeater.subscribe(self_filter_module)  # loopback in this ASR->TTS example

self_filter_module.subscribe(text_callback_module)


##################start everyone
speaker_module.run()
self_filter_module.run()
text_callback_module.run()
tts_module.run()
asr_module.run()
loopback_repeater.run()
microphone_module.run()
# speaking_trigger.run()

print("go")
input()   # wait for user key

#clean up and stop
microphone_module.stop()
asr_module.stop()
tts_module.stop()
speaker_module.stop()
loopback_repeater.stop()