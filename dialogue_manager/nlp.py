import re

import retico_core
from retico_core.abstract import AbstractModule
from retico_core.text import TextIU


class TextTokenizer(AbstractModule):
    @staticmethod
    def name():
        return "Text Tokenizer"

    @staticmethod
    def description():
        return "Tokenize the word"

    @staticmethod
    def input_ius():
        return [TextIU]

    @staticmethod
    def output_iu():
        return TextIU

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.tokens = []

    # ADD: tokenize every new word right away (includes guesses later revoked)
    def process_update(self, update_message):
        um = retico_core.UpdateMessage()
        for iu, ut in update_message:
            if ut == retico_core.UpdateType.ADD:
                words = re.findall(r"\w+", iu.text.lower())
                self.tokens.extend(words)
                out = self.create_iu(grounded_in=iu)
                out.payload = words
                um.add_iu(out, retico_core.UpdateType.ADD)
        return um if len(um) > 0 else None

    # # COMMIT: tokenize only final words (slower, but no wrong guesses)
    # def process_update(self, update_message):
    #     um = retico_core.UpdateMessage()
    #     for iu, ut in update_message:
    #         if ut == retico_core.UpdateType.COMMIT:
    #             words = re.findall(r"\w+", iu.text.lower())
    #             self.tokens.extend(words)
    #             out = self.create_iu(grounded_in=iu)
    #             out.payload = words
    #             um.add_iu(out, retico_core.UpdateType.ADD)
    #     return um if len(um) > 0 else None
