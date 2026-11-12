from typesafe_sdk import Choice, TypeSafeClient

MODEL = "jev-1.13.0"

QUESTIONS = {
    "stance": Choice(
        instructions="What is the participant's stance toward the robot in `utterance`? Interpret it in light of `previous_turns` (the current round's conversation) and `game_history` (what both players did in earlier rounds), especially when judging sarcasm, accusations, or apologies.",
        criteria={
            "favor": "Aligns or collaborates with the robot: agrees with it, gives reasons or evidence for both staying silent, expresses trust, promises cooperation, or apologizes to repair the relationship.",
            "against": "Contrasts with the robot: disagrees with it, doubts its future move, argues for betraying, accuses it of betrayal or lying, insults it, or is sarcastic about its behavior.",
            "neither": "Takes no position with or against the robot: asks a neutral question, comments on the game or its outcome, small talk, technical issues, or questions to the experimenter.",
        },
    ),
    "intention": Choice(
        instructions="What does the player say they will do in the current `round`, in `utterance`? Use `previous_turns` only to interpret short replies like 'yes' or 'okay' (e.g., answering Sota's 'Can I count on you?'). Use `game_history` only to understand references to earlier rounds, not to guess the player's move.",
        criteria={
            "cooperate": "States, promises, or clearly implies they will cooperate this round.",
            "defect": "States, threatens, or clearly implies they will defect this round.",
            "undecided": "Expresses uncertainty, avoids committing, or has not said anything about their move.",
        },
    ),
}

_client = TypeSafeClient(model=MODEL)  # reads TYPESAFE_API_KEY


def classify(utterance, previous_turns=None, game_history=None, round_num=1):
    """Classify one user utterance. Returns the six probabilities and the model."""
    state = {
        "round": round_num,
        "game_history": game_history or [],
        "previous_turns": previous_turns or [],
        "utterance": f"Player: {utterance}",
    }
    response = _client.system_one(state=state, questions=QUESTIONS)
    stance = response.choices["stance"].probabilities
    intention = response.choices["intention"].probabilities
    return {
        "p_favor": stance["favor"],
        "p_against": stance["against"],
        "p_neither": stance["neither"],
        "p_cooperate": intention["cooperate"],
        "p_defect": intention["defect"],
        "p_undecided": intention["undecided"],
        "jev_model": MODEL,
    }
