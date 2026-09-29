import time

import retico_core
from retico_core.abstract import AbstractModule, AbstractProducingModule, UpdateMessage
from retico_core.text import TextIU

states = {"robot_talking": False}


class FakeMic(AbstractProducingModule):
    """Sends one word every 0.5 s, like the mic sending chunks.
    Each word also says whether the robot was talking at that moment."""

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
        time.sleep(0.5)
        if not self.words:
            return None
        word, robot_talking = self.words.pop(0)
        states["robot_talking"] = (
            robot_talking  # stands in for RobotTalking (the writer)
        )
        iu = self.create_iu()
        iu.payload = word
        return UpdateMessage.from_iu(iu, retico_core.UpdateType.ADD)


class TurnTagger(AbstractModule):
    """Check-in: tags each bag with whose turn it is."""

    @staticmethod
    def name():
        return "Turn Tagger"

    @staticmethod
    def description():
        return "Tags each IU with the current turn"

    @staticmethod
    def input_ius():
        return [TextIU]

    @staticmethod
    def output_iu():
        return TextIU

    def process_update(self, update_msg):
        tag = "robot" if states["robot_talking"] else "user"
        for iu, ut in update_msg:
            iu.meta_data["turn"] = tag
        return update_msg


class Sorter(AbstractModule):
    """Belt entrance: lets through only bags with its own tag."""

    @staticmethod
    def name():
        return "Sorter"

    @staticmethod
    def description():
        return "Passes only IUs with a given turn tag"

    @staticmethod
    def input_ius():
        return [TextIU]

    @staticmethod
    def output_iu():
        return TextIU

    def __init__(self, keep, **kwargs):
        super().__init__(**kwargs)
        self.keep = keep  # "user" or "robot"

    def process_update(self, update_msg):
        out = UpdateMessage()
        for iu, ut in update_msg:
            if iu.meta_data.get("turn") == self.keep:
                out.add_iu(iu, ut)
        return out if len(out) else None  # nothing for this belt → absorbed


def belt(name):
    def show(update_msg):
        for iu, ut in update_msg:
            print(f"[{name} belt] {iu.text!r:16} turn={iu.meta_data['turn']}")

    return show


# modules
mic = FakeMic(
    [
        ("hello", False),
        ("sota", False),
        ("how", False),
        ("(robot voice)", True),  # robot talking → echo
        ("(robot voice)", True),
        ("fine", False),
    ]
)
tagger = TurnTagger()
user_sorter = Sorter("user")
robot_sorter = Sorter("robot")
user_belt = retico_core.debug.CallbackModule(callback=belt("user"))
robot_belt = retico_core.debug.CallbackModule(callback=belt("robot"))

# conveyor
mic.subscribe(tagger)
tagger.subscribe(user_sorter)
tagger.subscribe(robot_sorter)
user_sorter.subscribe(user_belt)
robot_sorter.subscribe(robot_belt)

for m in [user_belt, robot_belt, user_sorter, robot_sorter, tagger, mic]:
    m.run()

time.sleep(4)  # let all words go through

for m in [mic, tagger, user_sorter, robot_sorter, user_belt, robot_belt]:
    m.stop()
