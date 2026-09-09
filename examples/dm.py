import Levenshtein
from retico_core import AbstractModule, UpdateMessage, UpdateType
from retico_core import abstract
from retico_core.text import SpeechRecognitionIU
from retico_core.text import TextIU
from retico_huggingfacelm import HuggingfaceLMClient


# A sample custom dialog manager
class TemplateDialogManagerModule(AbstractModule):

    def __init__(
            self,
            **kwargs):
        super().__init__(**kwargs)

    @staticmethod
    def name():
        return "Simple Dialog Manager Sample"

    @staticmethod
    def description():
        return "A simple example dialog manager."

    @staticmethod
    def input_ius():
        # you could also take in, e.g., audioIUs from the TTS to know when it is speaking to intelligently interrupt
        return [SpeechRecognitionIU, TextIU]  # accept input from speech recognition directly, or any text source

    @staticmethod
    def output_iu():
        return TextIU

    def process_update(self, update_message):
        output_update = UpdateMessage()
        has_output_ius = False

        for iu, ut in update_message:

            if ut == abstract.UpdateType.ADD:
                # we can do something here as words come in but aren't committed, like, react to keywords?
                # or if the person is speaking, pay attention or act accordingly.
                # could also link the actual speaking input (pre-ASR) to get even quicker data
                pass

            elif ut == abstract.UpdateType.REVOKE:
                # we can do something when the ASR revokes candidates. Like, if we had a keyword we reacted to but
                # it got revoked we could over-compensate a reaction, etc.
                pass

            elif ut == abstract.UpdateType.COMMIT:

                output_iu = self.create_iu(iu)
                # output_iu.set_audio(iu.raw_audio, iu.nframes, iu.rate, iu.sample_width)
                output_update.add_iu(output_iu, ut)
                has_output_ius = True

        if has_output_ius:
            self.append(output_update)
