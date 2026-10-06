import joblib
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# ---------------------------------------------------------------
# Models
# ---------------------------------------------------------------
dd = joblib.load("DAILYDIALOG/dailydialog_models.joblib")

analyzer = SentimentIntensityAnalyzer()
# Game words (scale: -4 very negative ... +4 very positive)
# Only words VADER lacks, plus "defect" made stronger (in the game it means betrayal)
GAME_WORDS = {
    # cooperation (missing from VADER)
    "cooperate": 1.5,
    "cooperates": 1.5,
    "cooperated": 1.5,
    "cooperating": 1.5,
    "cooperation": 1.5,
    "cooperative": 1.5,
    # alliances and teamwork (missing from VADER)
    "ally": 1.5,
    "allies": 1.5,
    "alliance": 1.5,
    "partner": 1.0,
    "partners": 1.0,
    "team": 1.0,
    "teammate": 1.0,
    "together": 1.0,
    "deal": 1.0,
    # defection (VADER has -1.4 to -1.8; stronger here)
    "defect": -2.5,
    "defects": -2.5,
    "defected": -2.5,
    "defecting": -2.5,
    "defection": -2.5,
    # betrayal and deception (missing from VADER)
    "traitor": -2.8,
    "backstab": -3.0,
    "backstabbed": -3.0,
    "backstabber": -3.0,
    "backstabbing": -3.0,
    "lie": -1.8,
    "lies": -1.8,
    "snitch": -2.0,
    "mistrust": -1.8,
    "disloyal": -2.3,
    "stole": -2.2,
    "retaliate": -1.8,
}
analyzer.lexicon.update(GAME_WORDS)

# ---------------------------------------------------------------
# Settings
# ---------------------------------------------------------------
INTENT_WEIGHT = {"inform": 1.0, "directive": 0.7, "commissive": 0.7, "question": 0.5}
STANCES = [
    (0.50, "trusting"),
    (0.75, "suspicious"),
    (1.00, "angry / betrayed"),
]

INITIAL_STATE = 0.30  # start slightly trusting
RATE_DISTRUST = 0.9  # fast move toward distrust
RATE_TRUST = 0.3  # slow move back toward trust


# ---------------------------------------------------------------
# Functions (same as test_stance.py)
# ---------------------------------------------------------------
def fix_text(text):
    for bad in ["\u2019", "\u2018", "\u00e2\u0080\u0099", "\x92", "\u00e2\u20ac\u2122"]:
        text = text.replace(bad, "'")
    return text


def sentiment_score(text):
    compound = analyzer.polarity_scores(text)["compound"]
    return (1 - compound) / 2


def intent_of(text):
    X = dd["vectorizer"].transform([fix_text(text)])
    model = dd["intent_model"]
    probs = pd.Series(model.predict_proba(X)[0], index=model.classes_)
    return str(probs.idxmax())


def utterance_stance(text):
    sentiment = sentiment_score(text)
    intent = intent_of(text)
    score = 0.5 + INTENT_WEIGHT[intent] * (sentiment - 0.5)
    return intent, score


def band(score):
    return next(name for limit, name in STANCES if score <= limit)


# ---------------------------------------------------------------
# New: update the state turn by turn
# ---------------------------------------------------------------
def update_state(state, stance_now):
    rate = RATE_DISTRUST if stance_now > state else RATE_TRUST
    return state + rate * (stance_now - state)


def run_conversation(title, utterances):
    print(f"\n=== {title} ===")
    print(
        f"{'turn':4s} {'utterance':40s} {'intent':11s} {'now':>5s} {'state':>6s}  band"
    )
    state = INITIAL_STATE
    for turn, text in enumerate(utterances, start=1):
        intent, stance_now = utterance_stance(text)
        state = update_state(state, stance_now)
        print(
            f"{turn:<4d} {text:40s} {intent:11s} {stance_now:5.2f} {state:6.2f}  {band(state)}"
        )


# ---------------------------------------------------------------
# Test conversations
# ---------------------------------------------------------------
run_conversation(
    "Trust, then sudden betrayal, then recovery",
    [
        "Let's both cooperate.",
        "I promise I will cooperate this time.",
        "I'm so happy we won together!",
        "You betrayed me!",
        "Okay, let's cooperate again.",
        "I trust you this time.",
        "Great, we both won!",
    ],
)

run_conversation(
    "Growing suspicion",
    [
        "Let's both cooperate.",
        "Will you betray me?",
        "I think you will defect.",
        "Why did you defect?",
    ],
)
