from sota_thinclient import ConnectionManager
from sota_retico import SotaMicrophoneModule
from sota_retico.sota_audio import SotaSpeakerModule, AudioGatingModule, SpeakerTrigger
from retico_whisperasr import WhisperASRModule
from retico_speechbraintts import  SpeechBrainTTSModule
import retico_core
# from filter import RobotASRFilterModule, SimpleTextPassthrough



# SOTA_IP = "192.168.0.23"
SOTA_IP = "10.151.63.71"
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


sota = ConnectionManager(SOTA_IP, HTTP_PORT)

#################initialize the retico modules
speaking_trigger = SpeakerTrigger()

microphone_module = SotaMicrophoneModule(sota, MIC_UDP_PORT, buffer_ms=20)
speaker_module = SotaSpeakerModule(sota, SPEAKER_UDP_PORT, speaking_trigger=speaking_trigger)
text_callback_module = retico_core.debug.CallbackModule(callback=text_candidate_callback)

print("Starting Whisper ASR...", end="")
asr_module = WhisperASRModule()
print("done.")

print("Starting SpeechBrains...", end="")
tts_module = SpeechBrainTTSModule(language="en")
print("done.")

self_filter_module = AudioGatingModule(sleep_interval=0.001)


#################setup the connections
microphone_module.subscribe(self_filter_module)
self_filter_module.subscribe(asr_module)
asr_module.subscribe(tts_module)
asr_module.subscribe(text_callback_module)
tts_module.subscribe(speaker_module)
speaking_trigger.subscribe(self_filter_module)

##################start everyone
speaker_module.run()
self_filter_module.run()
text_callback_module.run()
tts_module.run()
asr_module.run()
microphone_module.run()
speaking_trigger.run()

print("go")
input()   # wait for user key

#clean up and stop
microphone_module.stop()
asr_module.stop()
tts_module.stop()
speaker_module.stop()