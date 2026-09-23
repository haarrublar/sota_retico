### A debugging callback function that retico can send data to.
# here we use it as a way to see what the ASR is picking up. It's interesting and useful for debugging.
from random import sample

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


import numpy as np

recording = False
sample_rate = None
scut = 1.0
max_bytes = 2**16
audio_info = []
wav_array = np.array([], dtype=np.int16)
wav_data = np.array([], dtype=np.int16)


def audio_array_callback(update_msg):
    """Collects microphone audio into 1-second segments and measures their loudness.

    Called by retico's CallbackModule each time the microphone sends audio.
    While `recording` is True, it adds each chunk to `wav_array`. When `wav_array`
    reaches `scut` seconds, it stores the segment in `audio_info` together with its
    rate, length and RMS loudness, then starts a new segment.

    Args:
        update_msg: a retico UpdateMessage of (AudioIU, UpdateType) pairs.
    """
    global wav_array, sample_rate

    if not recording:
        return

    for x, ut in update_msg:
        if ut == retico_core.UpdateType.ADD:
            samples = np.frombuffer(x.raw_audio, dtype=np.int16)
            wav_array = np.concatenate([wav_array, samples], dtype=np.int16)

            if sample_rate is None:
                sample_rate = x.rate
            thres = len(wav_array) / sample_rate
            if thres >= scut:
                norm = wav_array / max_bytes
                audio_info.append(
                    {
                        "rate": sample_rate,
                        "audio": wav_array,
                        "seconds": thres,
                        "rms": np.sqrt(np.mean(norm**2)),
                    }
                )
                wav_array = np.array([], dtype=np.int16)


def close_wav():
    global wav_file
    if wav_file is not None:
        wav_file.close()
        wav_file = None
