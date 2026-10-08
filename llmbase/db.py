from retico_core import AbstractModule, UpdateMessage, abstract
from retico_core.text import SpeechRecognitionIU, TextIU
from sqlalchemy import Column, Integer, String, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

Base = declarative_base()
engine = create_engine("sqlite:///conversation.db")


class DBConversation(Base):
    __tablename__ = "conversation"
    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String)
    utterance = Column(String)


Base.metadata.create_all(engine)  # creates the table the first time
Session = sessionmaker(bind=engine)  # made once, not on every update


class DBModule(AbstractModule):
    def __init__(self, llm_module, **kwargs):
        super().__init__(**kwargs)
        self.llm = llm_module
        self.user_ius = []  # words of the sentence I am saying
        self.sota_ius = []  # words of sota's reply

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

    def _saveDB(self, name, ius):
        text = " ".join(iu.text for iu in ius).strip()
        if not text:
            return
        with Session() as session:
            session.add(DBConversation(name=name, utterance=text))
            session.commit()

    def process_update(self, update_message):
        for iu, ut in update_message:
            is_user = isinstance(iu, SpeechRecognitionIU)
            ius = self.user_ius if is_user else self.sota_ius
            if ut == abstract.UpdateType.ADD:
                ius.append(iu)
            if ut == abstract.UpdateType.REVOKE:
                ius.remove(iu)
            if ut == abstract.UpdateType.COMMIT and is_user:
                self._saveDB("user", self.user_ius)  # my sentence is finished
                self.user_ius = []

        if self.sota_ius and self.llm.reply_done:
            self._saveDB("sota", self.sota_ius)
            self.sota_ius = []

    def shutdown(self):
        # save what is left when I press Enter
        self._saveDB("user", self.user_ius)
        self._saveDB("sota", self.sota_ius)
