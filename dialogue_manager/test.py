from audioManager import AudioAnnotator, AudioClassifier, SilenceRemover, WavFile
from sota_audio import SotaMicrophoneModule
from userMicrophoneModule import UserMicrophoneModule

from sota_thinclient import ConnectionManager

SOTA_IP = "10.151.63.79"
HTTP_PORT = "8080"
MIC_UDP_PORT = 52001

user_mic = UserMicrophoneModule()
sota = ConnectionManager(SOTA_IP, HTTP_PORT)
sota_mic = SotaMicrophoneModule(sota, MIC_UDP_PORT, buffer_ms=20)
audio_annotator = AudioAnnotator()
silence_filter = SilenceRemover()
# user_wav = WavFile("user.wav")
sota_wav = WavFile("sota.wav")
sota_audio = AudioClassifier(keep="sota")
# user_audio = AudioClassifier(keep="user")


# subscribing modules
# user_mic.subscribe(audio_annotator)
# audio_annotator.subscribe(user_audio)
sota_mic.subscribe(audio_annotator)
audio_annotator.subscribe(silence_filter)
silence_filter.subscribe(sota_audio)
sota_audio.subscribe(sota_wav)


modules = [sota_wav, sota_audio, silence_filter, audio_annotator, sota_mic]
for m in modules:  # receivers first, mics last
    m.run()

input("running... press Enter to stop\n")

for m in reversed(modules):  # mics first
    m.stop()
