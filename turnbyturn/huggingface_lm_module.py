import time  # (for the gate)

import retico_core

# from retico_huggingfacelm.huggingface_lm_client import HuggingfaceLMClient
from huggingface_lm_client import HuggingfaceLMClient
from retico_core import abstract
from retico_core.text import SpeechRecognitionIU, TextIU


class HuggingfaceLMModule(abstract.AbstractModule):
    def __init__(self, lm_client: HuggingfaceLMClient):
        super().__init__()
        self.lm_client = lm_client
        self.reply_done = True  #  false while sota's reply is being written
        self.done_at = 0.0  # time when the last reply finished
        self._last_iu = None  # last word sent, the commit points to it
        self._pending = 0  # words sent since the last commit

    @staticmethod
    def name():
        return "Hugging Face LM Module"

    @staticmethod
    def description():
        return "A module running Hugging Face language model for real-time dialogue."

    @staticmethod
    def input_ius():
        return SpeechRecognitionIU

    @staticmethod
    def output_iu():
        return TextIU

    def process_update(self, update_message):
        send_prompt = False

        for iu, ut in update_message:
            if ut == abstract.UpdateType.ADD:
                # self.current_output.append(iu)
                self.current_input.append(iu)  # the user's words are input, not output
            elif ut == abstract.UpdateType.REVOKE:
                # self.revoke(iu)
                if iu in self.current_input:  # sends a revoke to the tts
                    self.current_input.remove(iu)
            elif ut == abstract.UpdateType.COMMIT:
                send_prompt = True

        if send_prompt:
            send_prompt = False
            last_commit_sentence = ""
            # for unit in self.current_output:
            for unit in self.current_input:
                last_commit_sentence += f"{unit.text} "
            # self.current_output = []
            self.current_input = []

            # if len(last_commit_sentence) > 0:
            if last_commit_sentence.strip():  # changed: ignore commits with no words
                # print('user:', last_commit_sentence)
                self.generate_model_output(last_commit_sentence)

    def generate_model_output(self, last_commit_sentence):

        ## waits for the whole reply, then sends all the words with one commit at the end
        # response = self.lm_client.generate_response(last_commit_sentence)
        # words = response.split()
        #
        # current_iu = None
        # for word in words:
        #     current_iu = self.create_iu()
        #     current_iu.payload = word
        #     update_message = retico_core.UpdateMessage.from_iu(current_iu, retico_core.UpdateType.ADD)
        #     self.append(update_message)
        #
        # # Send singular COMMIT to signal end of output/response for a given prompt
        # if current_iu is not None:
        #     update_message = retico_core.UpdateMessage.from_iu(current_iu, retico_core.UpdateType.COMMIT)
        #     self.append(update_message)

        # send the words while the model writes them
        self.reply_done = False
        self._last_iu = None
        self._pending = 0
        buffer = ""
        try:
            for piece in self.lm_client.stream_response(last_commit_sentence):
                # piper reads * and # out loud, so I remove them
                buffer += piece.replace("*", "").replace("#", "").replace("\n", " ")
                # the last part can be half a word, so it waits in the buffer
                *words, buffer = buffer.split(" ")
                for word in words:
                    self._send_word(word)
            self._send_word(buffer)
            if self._pending:  # last words without a commit
                self._commit()
        finally:
            # also if something fails, so the gate does not stay closed
            self.reply_done = True
            self.done_at = time.time()

    # added: aux function, sends one word to the tts
    def _send_word(self, word):
        word = word.strip()
        if not word:
            return
        current_iu = self.create_iu()
        current_iu.payload = word
        update_message = retico_core.UpdateMessage.from_iu(
            current_iu, retico_core.UpdateType.ADD
        )
        self.append(update_message)
        self._last_iu, self._pending = current_iu, self._pending + 1

        # commit at the end of a sentence, a comma, or with 8 words waiting
        # (with only . ! ? a long sentence made sota wait almost all the reply)
        if word[-1] in ".!?,;:" or self._pending >= 8:
            self._commit()

    # aux function, tells the tts to speak the words it has now
    def _commit(self):
        update_message = retico_core.UpdateMessage.from_iu(
            self._last_iu, retico_core.UpdateType.COMMIT
        )
        self.append(update_message)
        self._pending = 0

    def process_revoke(self, iu):
        pass
