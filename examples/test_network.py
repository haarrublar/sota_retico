from retico_core import AbstractModule, UpdateMessage, UpdateType, network
from retico_core.audio import MicrophoneModule, SpeakerModule
from retico_core.debug import CallbackModule
from retico_core.text import SpeechRecognitionIU, TextIU
from retico_googleasr import GoogleASRModule
from retico_mistyrobot import misty_action
from retico_opendialdm.dm import OpenDialModule, DialogueActIU, DialogueDecisionIU
from retico_speechbraintts import SpeechBrainTTSModule

import os
import re
import time

from dotenv import load_dotenv
load_dotenv()
os.environ['PYOD']="./pyopendial/"

opendial_variables = ['firstname', 'lastname', 'work', 'email', 'notes']
domain_dir = 'dialogue.xml'

class SimpleNLUModule(AbstractModule):
    def __init__(self):
        super().__init__()
        self.word_buffer = []
        
    @staticmethod
    def name():
        return "Simple NLU Module"

    @staticmethod
    def description():
        return "A module that performs simple NLU."

    @staticmethod
    def input_ius():
        return [SpeechRecognitionIU, TextIU]

    @staticmethod
    def output_iu():
        return DialogueActIU
    
    def process_update(self, update: UpdateMessage):
        for iu, _ in update:
            word = iu.payload
            if word:
                self.word_buffer.append(word)
            if iu.final:
                utterance = ' '.join(self.word_buffer).lower()
                self.word_buffer.clear()
            else:
                utterance = ' '.join(self.word_buffer).lower()
            print(f"[DEBUG] Current utterance: {utterance}")
            payload = {}
            # First name
            fn_match = re.search(r"first name is ([a-zA-Z'-]+)", utterance)
            if fn_match:
                payload['firstname'] = fn_match.group(1)
            # Last name
            ln_match = re.search(r"last name is ([a-zA-Z'-]+)", utterance)
            if ln_match:
                payload['lastname'] = ln_match.group(1)
            # Work
            work_match = re.search(r"work(s|ed|ing)? (at|in) ([a-zA-Z0-9' -]+)", utterance)
            if work_match:
                payload['work'] = work_match.group(3)
            # Email
            utterance = utterance.lower()

            email_match = re.search(
                r"email is\s+([\w\s\.\-]+\s*(?:@|at)\s*[\w\.-]+\s*(?:\.|dot)\s*[a-zA-Z]{2,3})",
                utterance
            )
            if email_match:
                payload["email"] = email_match.group(1).strip().replace(' ', '').replace('at', '@').replace('dot', '.')
            # Note
            note_match = re.search(r"(?:well|um|so|yep|yeah|yes|add|say|tell) ([a-zA-Z0-9' -]+)", utterance)
            if note_match:
                payload['notes'] = note_match.group(1)
        iu_out = DialogueActIU(payload=payload)
        
        um = UpdateMessage()
        um.add_iu(iu_out, UpdateType.ADD)
        self.append(um)

class ResponseModule(AbstractModule):
    def __init__(self):
        super().__init__()
        
    responses = {
        'ask_about_firstname': "What is the person's first name?",
        'ask_about_lastname': "What is {firstname}'s last name?",
        'ask_about_work': "Where does {firstname} {lastname} work?",
        'ask_about_email': "What is {firstname} {lastname}'s email?",
        'ask_about_notes': "Do you have a note to add about {firstname} {lastname}?",
        'all_slots_filled': "That's all I need for this person! Let's recap: Name: {firstname} {lastname}. Work: {work}. Email: {email}. Notes: {notes}. Is that correct?",
    }
    
    @staticmethod
    def name():
        return "Response Module"

    @staticmethod
    def description():
        return "A module that generates responses based on DM decisions."

    @staticmethod
    def input_ius():
        return [DialogueDecisionIU]

    @staticmethod
    def output_iu():
        return TextIU
    
    def process_update(self, update: UpdateMessage):
        for iu, _ in update:
            decision = iu.payload.get('decision', None)
            dialogue_vars = iu.payload.get('concepts', {})
            if decision is None:
                print(f"[WARNING] No 'decision' in DM output payload: {iu.payload}")
            response = self.responses.get(decision, "I didn't understand, could you please repeat?")
            for var in dialogue_vars: # Template-based response generation
                if dialogue_vars[var] is not None:
                    response = response.format(**dialogue_vars)
            
            iu_out = TextIU(payload=response, iuid=0)
            um = UpdateMessage()
            um.add_iu(iu_out, UpdateType.ADD)
            um.add_iu(iu=TextIU(payload='', iuid=hash(time.time())), update_type=UpdateType.COMMIT)
            self.append(um)

def callback(update_msg):
    for iu, ut in update_msg:
        text = getattr(iu, 'text', iu.payload if hasattr(iu, 'payload') else None)
        print(f"{ut}: {text}")

if __name__ == "__main__":

    # Websocket-based approach (Misty only)
    
    # from retico_robot_filter import MistyRobotASRFilterModule

    # mic = MicrophoneModule()
    # asr = GoogleASRModule(language="en-US")
    # nlu = SimpleNLUModule()
    # dm = OpenDialModule(domain_dir=domain_dir, variables=opendial_variables)
    # nlg = misty_action.MistyActionModule(ip='10.10.0.3', volume=15)
    # tts = SpeechBrainTTSModule()
    # spk = SpeakerModule(rate=22050, volume=0.5)
    # debug = CallbackModule(callback)
    
    # fil = MistyRobotASRFilterModule(action_module=nlg)

    # mic.subscribe(asr)
    # asr.subscribe(fil)
    # fil.subscribe(nlu)
    # nlu.subscribe(dm)
    # dm.subscribe(nlg)
    # nlg.subscribe(tts)
    # tts.subscribe(spk)
    
    # asr.subscribe(debug)
    # nlu.subscribe(debug)

    # network.run(mic)

    # input("Running...\n")

    # network.stop(mic)


    # -------------------------------- #
    
    
    # Similarity-based approach
    
    from retico_robot_filter import RobotASRFilterModule
    
    mic = MicrophoneModule()
    asr = GoogleASRModule(language="en-US")
    fil = RobotASRFilterModule()
    nlu = SimpleNLUModule()
    dm = OpenDialModule(domain_dir=domain_dir, variables=opendial_variables)
    nlg = misty_action.MistyActionModule(ip='10.10.0.3', volume=30)
    tts = SpeechBrainTTSModule()
    spk = SpeakerModule(rate=22050, volume=0.5)
    debug = CallbackModule(callback)
    
    mic.subscribe(asr)
    asr.subscribe(fil)
    fil.subscribe(nlu)
    nlu.subscribe(dm)
    dm.subscribe(nlg)
    nlg.subscribe(fil)
    
    asr.subscribe(debug)
    nlu.subscribe(debug)
    
    network.run(mic)
    
    input("Running...\n")

    network.stop(mic)