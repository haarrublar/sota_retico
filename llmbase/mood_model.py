from typesafe_sdk import Choice, TypeSafeClient

MODEL = "jev-1.13.0"

# Values from the markdown
V_FAVOR = 0.25  # favor
V_AGAINST_NB = 0.50  # against, no betrayals
V_AGAINST_B = 0.75  # against, only betrayals
ALPHA_UP = 0.45  # toward negative
ALPHA_DOWN = 0.17  # toward positive
TRUST_THRESHOLD = 0.55  # Sota cooperates only above this
PROMISE_THRESHOLD = 0.70  # p(cooperate) needed to count as a promise

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

client = TypeSafeClient(model=MODEL)  # reads TYPESAFE_API_KEY


def classify(state):
    """Send one player turn to Jev and return the two probability dicts."""
    response = client.system_one(state=state, questions=QUESTIONS)
    stance = response.choices["stance"]
    intention = response.choices["intention"]
    return stance.probabilities, intention.probabilities


class SotaMood:
    def __init__(self, valence=0.30, kept=2, broken=0):
        self.mood = valence  # valence: 0 positive, 1 negative
        self.kept = kept  # memory: promises kept
        self.broken = broken  # memory: promises broken
        self.history = []  # game_history sent to Jev

    def broken_promise(self):
        """Share of broken promises."""
        return self.broken / (self.kept + self.broken)

    def trust(self):
        return 1 - self.broken_promise()

    def target(self, p):
        """Section 7: expected value, weighted by memory."""
        b = self.broken_promise()
        v_against = (1 - b) * V_AGAINST_NB + b * V_AGAINST_B
        return (
            (1 - b) * p["favor"] * V_FAVOR
            + p["against"] * v_against
            + (p["neither"] + b * p["favor"]) * self.mood
        )

    def update(self, p):
        """Euler step with asymmetric rates."""
        T = self.target(p)
        alpha = ALPHA_UP if T > self.mood else ALPHA_DOWN
        self.mood = self.mood + alpha * (T - self.mood)
        return T

    def band(self):
        return "positive" if self.mood <= 0.50 else "negative"

    def decide(self, promised):
        """Section 9."""
        return "silent" if self.trust() >= TRUST_THRESHOLD and promised else "betray"

    def reveal(self, round_num, promised, sota_move, player_move):
        """Section 8: memory update after the decision."""
        if promised:
            if player_move == "silent":
                self.kept += 1
                self.mood += ALPHA_DOWN * (V_FAVOR - self.mood)  # kept promise
            else:
                self.broken += 1
                self.mood += ALPHA_UP * (V_AGAINST_B - self.mood)  # broken promise
        self.history.append(
            f"Round {round_num}: "
            f"{'The player promised to stay silent' if promised else 'The player made no promise'}, "
            f"then chose {player_move}. Sota chose {sota_move}."
        )


def analyze(sota, sentence, round_num=1, previous_turns=None):
    """Classify one player sentence and return all model variables."""
    state = {
        "round": round_num,
        "game_history": sota.history,
        "previous_turns": previous_turns or [],
        "utterance": f"Player: {sentence}",
    }
    mood_before = sota.mood
    p, intent = classify(state)
    T = sota.update(p)
    b = sota.broken_promise()
    return {
        "sentence": sentence,
        "p_favor": p["favor"],
        "p_against": p["against"],
        "p_neither": p["neither"],
        "p_cooperate": intent["cooperate"],
        "p_defect": intent["defect"],
        "p_undecided": intent["undecided"],
        "promise": intent["cooperate"] >= PROMISE_THRESHOLD,
        "broken_share": b,
        "trust": sota.trust(),
        "v_against": (1 - b) * V_AGAINST_NB + b * V_AGAINST_B,
        "target": T,
        "alpha": ALPHA_UP if T > mood_before else ALPHA_DOWN,
        "valence_before": mood_before,
        "valence_after": sota.mood,
        "band": sota.band(),
    }


def play_round(sota, round_num, dialogue, player_move):
    """dialogue: list of (speaker, text). Classifies every player line."""
    print(
        f"\n=== Round {round_num} | valence {sota.mood:.2f} ({sota.band()}) "
        f"| memory {sota.kept} kept / {sota.broken} broken | tau {sota.trust():.2f} ==="
    )
    turns, promised = [], False
    for speaker, text in dialogue:
        line = f"{speaker}: {text}"
        if speaker == "Player":
            state = {
                "round": round_num,
                "game_history": sota.history,
                "previous_turns": turns,
                "utterance": line,
            }
            p, intent = classify(state)
            T = sota.update(p)
            if intent["cooperate"] >= PROMISE_THRESHOLD:
                promised = True
            print(
                f"{text[:48]:48s} fav {p['favor']:.2f} ag {p['against']:.2f} "
                f"nei {p['neither']:.2f} | T {T:.2f} -> m {sota.mood:.2f} ({sota.band()})"
                f"{'  PROMISE' if intent['cooperate'] >= PROMISE_THRESHOLD else ''}"
            )
        turns.append(line)

    sota_move = sota.decide(promised)
    print(f"Sota decides: {sota_move} | player chooses: {player_move}")
    sota.reveal(round_num, promised, sota_move, player_move)
    print(
        f"Memory now: {sota.kept} kept / {sota.broken} broken | tau {sota.trust():.2f}"
    )


if __name__ == "__main__":
    sota = SotaMood()
    result = analyze(sota, "I promise, I'll stay silent.")
    for key, value in result.items():
        print(
            f"{key:15s} {value:.2f}"
            if isinstance(value, float)
            else f"{key:15s} {value}"
        )
