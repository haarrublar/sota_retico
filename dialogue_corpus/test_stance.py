import joblib
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# ---------------------------------------------------------------
# Models
# ---------------------------------------------------------------
dd = joblib.load("DAILYDIALOG/dailydialog_models.joblib")

analyzer = SentimentIntensityAnalyzer()

# Game words VADER does not know (scale: -4 very negative ... +4 very positive)
GAME_WORDS = {
    "cooperate": 1.5,
    "cooperation": 1.5,
    "cooperating": 1.5,
    "trust": 1.5,
    "defect": -2.0,
    "defected": -2.0,
    "defecting": -2.0,
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


# ---------------------------------------------------------------
# Functions
# ---------------------------------------------------------------
def fix_text(text):
    for bad in ["\u2019", "\u2018", "\u00e2\u0080\u0099", "\x92", "\u00e2\u20ac\u2122"]:
        text = text.replace(bad, "'")
    return text


def sentiment_score(text):
    compound = analyzer.polarity_scores(text)["compound"]  # -1 negative ... +1 positive
    return (1 - compound) / 2  # 0 = positive, 1 = negative


def intent_of(text):
    X = dd["vectorizer"].transform([fix_text(text)])
    model = dd["intent_model"]
    probs = pd.Series(model.predict_proba(X)[0], index=model.classes_)
    return str(probs.idxmax())


def stance(text):
    sentiment = sentiment_score(text)  # 1. how negative (0 to 1)
    intent = intent_of(text)  # 2. what the person is doing
    weight = INTENT_WEIGHT[intent]
    score = 0.5 + weight * (sentiment - 0.5)  # 3. pull toward 0.5 for indirect intents
    label = next(name for limit, name in STANCES if score <= limit)  # 4. name the band
    return intent, sentiment, score, label


# ---------------------------------------------------------------
# Test
# ---------------------------------------------------------------
sentences = [
    "Will you betray me?",
    "You betrayed me!",
    "I promise I will cooperate this time.",
    "Let's both cooperate.",
    "I'm so happy we won together!",
    "I think you will defect.",
]

for s in sentences:
    intent, sentiment, score, label = stance(s)
    print(
        f"{s:40s} intent={intent:11s} sentiment={sentiment:.2f} stance={score:.2f} -> {label}"
    )
