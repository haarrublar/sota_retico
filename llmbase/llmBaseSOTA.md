# Streaming the LLM reply so Sota talks sooner

## Summary

Before, Sota only started talking when the LLM finished the whole reply. Now the LLM sends the words to the TTS while it is still writing them, and the TTS speaks every small piece (sentence, comma or 8 words). So Sota starts talking after the first few words (about 1 s after I finish), and the gate still waits for the whole reply before opening.

## The problem

My pipeline is: Sota mic -> gate -> Whisper ASR -> LLM (Qwen 2.5 0.5B) -> Piper TTS -> Sota speaker.

With the normal module from `retico_huggingfacelm`, every turn was like this:

1. I finish talking, the ASR commits.
2. Qwen writes the whole reply (on my Mac CPU this is slow, around 10 tokens per second).
3. Only then the words go to the TTS.
4. Piper makes the audio of all the reply.
5. Sota starts talking.

So the waiting time is the time to write everything + the time to synthesize everything.

## The previous implementation

This is the part of the original module that sends the reply:

    def generate_model_output(self, last_commit_sentence):
        response = self.lm_client.generate_response(last_commit_sentence)
        words = response.split()

        current_iu = None
        for word in words:
            current_iu = self.create_iu()
            current_iu.payload = word
            update_message = retico_core.UpdateMessage.from_iu(current_iu, retico_core.UpdateType.ADD)
            self.append(update_message)

        if current_iu is not None:
            update_message = retico_core.UpdateMessage.from_iu(current_iu, retico_core.UpdateType.COMMIT)
            self.append(update_message)

`generate_response` blocks until the model finishes all the reply. After that the words are sent one by one, but all in the same moment, and there is only one COMMIT at the very end. The TTS speaks on COMMIT, so it waits for everything.

## What I changed

I modified the original `huggingface_lm_client.py` and `huggingface_lm_module.py`. The original lines are still there as comments, and the new parts are aux functions. I keep both files in my project folder and `main.py` imports them from there (the installed package is in site-packages, so changes there are not in git and an update erases them).

### 1. The client gives the text while it is written

- `_prepare_input` (new): the part of `generate_response` that adds the question to the history and tokenizes it. Now `generate_response` and `stream_response` use the same code, and `generate_response` works like before.
- `stream_response` (new): runs `model.generate` in a thread with a `TextIteratorStreamer` and gives the text piece by piece while the model writes it. It prints the reply in the terminal and keeps the history like `generate_response`.
- Small fix: in `set_max_history_turns`, `_messages_history` (does not exist) is now `_message_history`.

### 2. The module sends words as they arrive

- `generate_model_output`: the original lines are commented. Now it reads from `stream_response` and sends every word to the TTS right away.
- `_send_word` (new): sends one word, and a COMMIT at `.` `!` `?`, at a comma or `;` `:`, or when 8 words are waiting.
- `_commit` (new): sends the COMMIT so the TTS speaks what it has.
- A piece of text can end in half a word ("rob" + "ot"), so the last part waits in a buffer until the word is complete.
- I remove `*` and `#` because Piper reads them out loud.
- The user's words now go to `current_input` instead of `current_output`, and a revoke only removes the word (the original `self.revoke(iu)` was sending a revoke to the TTS).
- `reply_done` and `done_at` (new): tell the gate when the whole reply is finished.

### 3. The gate is now in sync with the LLM, the TTS and the speaker

With streaming, the reply arrives in pieces, and between two pieces there can be a pause while Qwen is still writing and Piper is still making the audio. During that pause the speaker is not sending and the mic is quiet, so the gate could think Sota finished and open, and then the ASR hears the next piece (echo).

So the gate now listens to every module that is part of Sota's turn:

| Module | Signal | What the gate does with it |
| --- | --- | --- |
| ASR | commit (`expect_reply`) | closes the gate, Sota's turn starts |
| LLM | `reply_done` and `done_at` | knows if the reply is still being written, and when it finished |
| TTS | (time to make the audio) | after the LLM finishes, waits 1.5 s more (`TTS_MARGIN`) so Piper can make the last piece |
| Speaker | `speaking` and `sent_seconds` | knows if audio is going out now, and how many seconds of speech Sota has to play |
| Sota mic | RMS of each chunk | knows when Sota's voice really reached the mic, and when it is quiet again |

The gate opens only when all of these are true:

1. The LLM finished the whole reply (`reply_done`) and 1.5 s passed since then.
2. The speaker is not sending anything.
3. Sota had time to play all the speech that was sent.
4. The mic was quiet for 0.6 s (30 chunks).

The pause between two pieces can't open it anymore, because point 1 is still false while Qwen is writing.

This is the new part in the gate:

    # the whole reply is written, and piper had time to make the last piece
    llm_done = self.llm is None or (
        self.llm.reply_done and now > self.llm.done_at + self.TTS_MARGIN
    )
    sota_finished = played_all and silence and not self.speaker.speaking and llm_done
    sota_silent = not heard and llm_done and now - self._closed_at > self.TIMEOUT

    if sota_finished or sota_silent:
        self._is_open = True

- `sota_silent` is a safety: if Sota never says anything (empty reply or an error), the gate opens after 15 s.
- `reply_done` is set to true in a `finally`, so even if the LLM fails the gate does not stay closed forever.
- `self.llm is None` means the gate still works without an LLM (the echo test).

A full turn now looks like this:

| Moment | LLM writing | Speaker sending | Mic | Gate |
| --- | --- | --- | --- | --- |
| I finish, ASR commits | starts | no | quiet | closes |
| first piece ready | yes | yes | Sota's voice | closed |
| pause between pieces | yes | no | quiet | stays closed (LLM not done) |
| last piece | finishes | yes | Sota's voice | closed |
| Sota finish + 0.6 s quiet | done | no | quiet | opens |

### 4. Less prints in the speaker

The `[SPEAKER] speech / silence` prints now only show with `debug=True`. With streaming they were printed for every piece and mixed with the LLM text in the terminal.

## Why commas were needed

My first version only committed at `.` `!` `?`. It didn't help much, because Qwen sometimes writes one long sentence with only commas:

> To ride your bike safely and efficiently, make sure you wear appropriate clothing, maintain proper pedaling technique, listen to music while riding, and adjust the handlebars as needed for comfort and performance. Enjoy your ride!

The first COMMIT came after around 40 words, almost at the end, so it looked the same as before. Committing at commas or every 8 words fixed it: here Sota starts after "To ride your bike safely and efficiently," (7 words).

## Timing

"Sota started" is the time from the ASR commit until Sota's voice reaches the mic (from the gate summary).

| Version | Reply | Sota started after commit |
| --- | --- | --- |
| Commit only at . ! ? | long sentence (bike) | 7.84 s |
| Commit at . ! ? , ; : or 8 words | short (how are you) | 1.64 s |
| Commit at . ! ? , ; : or 8 words | long, 19.4 s of speech (representation theory) | 1.05 s |

Before, the waiting time was the time to write the reply + synthesize the reply. Now it is the time to write the first few words + synthesize them, and the rest is prepared while Sota is already talking. Even with a 19 s reply Sota starts after about 1 s, and the gate opened 0.54 s after Sota finished.
