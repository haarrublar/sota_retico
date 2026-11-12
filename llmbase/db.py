from pathlib import Path

from jev_stance import classify
from retico_core import AbstractModule, abstract
from retico_core.text import SpeechRecognitionIU, TextIU
from sota_mood import PROMISE_THRESHOLD, SotaMood
from sqlalchemy import Boolean, Column, Float, Integer, String, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

DB_PATH = Path.home() / "sota_data" / "conversation.db"
DB_PATH.parent.mkdir(exist_ok=True)
engine = create_engine(f"sqlite:///{DB_PATH}")

Base = declarative_base()

CONTEXT_TURNS = 6


class DBConversation(Base):
    __tablename__ = "conversation"
    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String)
    utterance = Column(String)
    # Jev output (user rows)
    p_favor = Column(Float)
    p_against = Column(Float)
    p_neither = Column(Float)
    p_cooperate = Column(Float)
    p_defect = Column(Float)
    p_undecided = Column(Float)
    promise = Column(Boolean)
    jev_model = Column(String)
    # Mood (user rows: after the update; Sota rows: current value)
    target = Column(Float)
    alpha = Column(Float)
    valence_before = Column(Float)
    valence = Column(Float)
    band = Column(String)
    trust = Column(Float)


Base.metadata.create_all(engine)
Session = sessionmaker(bind=engine)


class DBModule(AbstractModule):
    def __init__(self, llm_module, **kwargs):
        super().__init__(**kwargs)
        self.llm = llm_module
        self.sota = SotaMood()
        self.user_ius = []
        self.sota_ius = []

    @staticmethod
    def name():
        return "Data base Management"

    @staticmethod
    def description():
        return "Storage Sota's and User's conversation"

    @staticmethod
    def input_ius():
        return [TextIU]

    @staticmethod
    def output_iu():
        return TextIU

    def _previous_turns(self, session):
        rows = (
            session.query(DBConversation)
            .order_by(DBConversation.id.desc())
            .limit(CONTEXT_TURNS)
            .all()
        )
        return [
            f"{'Player' if r.name == 'user' else 'Sota'}: {r.utterance}"
            for r in reversed(rows)
        ]

    def _saveDB(self, name, ius, classify_it=False):
        text = " ".join(iu.text for iu in ius).strip()
        if not text:
            return
        with Session() as session:
            row = DBConversation(name=name, utterance=text)

            if classify_it:
                labels = classify(
                    text,
                    previous_turns=self._previous_turns(session),
                    game_history=self.sota.history,
                )
                for key, value in labels.items():
                    setattr(row, key, value)
                row.promise = labels["p_cooperate"] >= PROMISE_THRESHOLD
                row.valence_before = self.sota.mood
                row.target, row.alpha = self.sota.update(
                    {
                        "favor": labels["p_favor"],
                        "against": labels["p_against"],
                        "neither": labels["p_neither"],
                    }
                )

            row.valence = self.sota.mood
            row.band = self.sota.band()
            row.trust = self.sota.trust()
            session.add(row)
            session.commit()

    def process_update(self, update_message):
        for iu, ut in update_message:
            is_user = isinstance(iu, SpeechRecognitionIU)
            ius = self.user_ius if is_user else self.sota_ius
            if ut == abstract.UpdateType.ADD:
                ius.append(iu)
            elif ut == abstract.UpdateType.REVOKE and iu in ius:
                ius.remove(iu)
            elif ut == abstract.UpdateType.COMMIT and is_user:
                self._saveDB("user", self.user_ius, classify_it=True)
                self.user_ius = []

        if self.sota_ius and self.llm.reply_done:
            self._saveDB("sota", self.sota_ius)
            self.sota_ius = []

    def shutdown(self):
        self._saveDB("user", self.user_ius)  # unfinished, not classified
        self._saveDB("sota", self.sota_ius)
