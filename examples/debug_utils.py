### A debugging callback function that retico can send data to.
# here we use it as a way to see what the ASR is picking up. It's interesting and useful for debugging.
import retico_core

msg = []


def text_candidate_callback(update_msg):
    global msg

    for x, ut in update_msg:  # update_msg is  (IU, IU type)
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
        print("\nCommitted: " + txt)

    else:
        print("\rlive: " + txt, end="")


import wave

wav_file = None


def audio_wavFile_callback(update_msg):
    """Saves the microphone audio to output.wav.

    Receives AudioIUs from the Sota microphone, opens the WAV file on the
    first chunk, and appends each new chunk. Call close_wav() when done (after stops).
    """
    global wav_file

    for x, ut in update_msg:
        if ut == retico_core.UpdateType.ADD:
            if wav_file is None:
                wav_file = wave.open("output.wav", "wb")
                wav_file.setnchannels(1)
                wav_file.setsampwidth(x.sample_width)
                wav_file.setframerate(x.rate)
            wav_file.writeframes(x.raw_audio)


def close_wav():
    global wav_file
    if wav_file is not None:
        wav_file.close()
        wav_file = None
