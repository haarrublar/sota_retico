import time

import retico_core
from retico_core.abstract import AbstractProducingModule, UpdateMessage
from retico_core.text import TextIU

WORDS = [
    "hello",
    "sota",
    "how",
    "are",
    "you",
    "doing",
    "today",
    "I",
    "am",
    "fine",
    "thank",
    "you",
    "for",
    "asking",
    "what",
    "is",
    "your",
    "name",
    "nice",
    "day",
]


class FakeMic(AbstractProducingModule):
    """Sends the words of a list, one every 0.3 s."""

    @staticmethod
    def name():
        return "Fake Mic"

    @staticmethod
    def description():
        return "Produces words from a list"

    @staticmethod
    def output_iu():
        return TextIU

    def __init__(self, words, **kwargs):
        super().__init__(**kwargs)
        self.words = list(words)

    def process_update(self, _):
        time.sleep(0.3)
        if not self.words:
            return None
        iu = self.create_iu()
        iu.payload = self.words.pop(0)
        return UpdateMessage.from_iu(iu, retico_core.UpdateType.ADD)


class TemplateType(retico_core.AbstractModule):
    """Gives every 5th word type 1, all others type 2."""

    @staticmethod
    def name():
        return "Template Type"

    @staticmethod
    def description():
        return "Types words: every 5th is type 1"

    @staticmethod
    def input_ius():
        return [TextIU]

    @staticmethod
    def output_iu():
        return TextIU

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.count = 0

    def process_update(self, update_message):
        for iu, ut in update_message:
            if ut == retico_core.UpdateType.ADD:
                self.count += 1
                iu.meta_data["type"] = 1 if self.count % 5 == 0 else 2
        return update_message


class TemplateOne(retico_core.AbstractModule):
    """Type 1 → remove_from_lb (cut the input). Everything else passes."""

    @staticmethod
    def name():
        return "Template One"

    @staticmethod
    def description():
        return "Cuts its input on a type 1 word, passes the rest"

    @staticmethod
    def input_ius():
        return [TextIU]

    @staticmethod
    def output_iu():
        return TextIU

    def process_update(self, update_message):
        out = retico_core.UpdateMessage()
        for iu, ut in update_message:
            if iu.meta_data.get("type") == 1:
                print(f"  {iu.text:7} type 1 -> removed")
                continue  # not added → the word is gone
            out.add_iu(iu, ut)  # type 2 → passes
        return out if len(out) else None  # nothing left → send nothing

    # def process_update(self, update_message):
    #     for iu, ut in update_message:
    #         if ut == retico_core.UpdateType.ADD and iu.meta_data.get("type") == 1:
    #             buffer = self.left_buffers()[0]  # the connection feeding us
    #             print(
    #                 f"  {iu.text:7} type 1 -> remove connection from {buffer.provider.name()}"
    #             )
    #             buffer.remove()  # what remove_from_lb does, without the bug
    #             return None
    #     return update_message  # type 2: pass


passed = []  # every IU that reached the end


def show(update_message):
    for iu, ut in update_message:
        passed.append(iu)
        print(f"  {iu.text:7} {iu.meta_data}  passed")


mic = FakeMic(WORDS)
typer = TemplateType()
one = TemplateOne()
out = retico_core.debug.CallbackModule(callback=show)

mic.subscribe(typer)
typer.subscribe(one)
one.subscribe(out)

for m in [out, one, typer, mic]:
    m.run()

time.sleep(len(WORDS) * 0.3 + 1)  # time for the whole list
print(
    f"end:  mic={mic._is_running}  typer={typer._is_running}  "
    f"one={one._is_running}  one inputs={len(one.left_buffers())}"
)

for m in [mic, typer, one, out]:
    m.stop()

print("\nfinal IUs:")
for iu in passed:
    print(f"  {iu.iuid:12} {iu.text:7} {iu.meta_data}")
print("\nfinal text:", " ".join(iu.text for iu in passed))
