import os
import urllib.request

import joblib
import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, CountVectorizer
from sklearn.metrics import classification_report
from sklearn.naive_bayes import MultinomialNB

# ---------------------------------------------------------------
# Settings
# ---------------------------------------------------------------
DATA_DIR = "MELD"
BASE_URL = "https://raw.githubusercontent.com/declare-lab/MELD/master/data/MELD/"
FILES = ["train_sent_emo.csv", "dev_sent_emo.csv", "test_sent_emo.csv"]
MODEL_PATH = os.path.join(DATA_DIR, "meld_models.joblib")

# Same emotion names as DailyDialog
MELD_TO_DAILYDIALOG = {"neutral": "no emotion", "joy": "happiness"}

KEEP = {
    "what",
    "where",
    "why",
    "how",
    "who",
    "whom",
    "whose",
    "when",
    "which",
    "will",
    "would",
    "can",
    "could",
    "should",
    "may",
    "might",
    "must",
    "please",
    "not",
    "no",
    "nor",
    "do",
    "does",
    "did",
}
STOP = (set(ENGLISH_STOP_WORDS) - KEEP) | set("abcdefghijklmnopqrstuvwxyz")

# ---------------------------------------------------------------
# Download the CSVs once (skipped if already in MELD/)
# ---------------------------------------------------------------
os.makedirs(DATA_DIR, exist_ok=True)
for name in FILES:
    path = os.path.join(DATA_DIR, name)
    if not os.path.exists(path):
        print("Downloading", name)
        urllib.request.urlretrieve(BASE_URL + name, path)


# ---------------------------------------------------------------
# Load and clean
# ---------------------------------------------------------------
def fix_text(text):
    # curly or broken apostrophes -> plain "'"
    for bad in ["\u2019", "\u2018", "\u00e2\u0080\u0099", "\x92", "\u00e2\u20ac\u2122"]:
        text = text.replace(bad, "'")
    return text


def load_meld(name):
    df = pd.read_csv(os.path.join(DATA_DIR, name), encoding="utf-8")
    return pd.DataFrame(
        {
            "dialogue_id": df["Dialogue_ID"],
            "utt_idx": df["Utterance_ID"],
            "utterance": df["Utterance"].astype(str).apply(fix_text),
            "emotion": df["Emotion"].replace(MELD_TO_DAILYDIALOG),
            "sentiment": df["Sentiment"],
        }
    )


train_utt = load_meld("train_sent_emo.csv")
val_utt = load_meld("dev_sent_emo.csv")

# ---------------------------------------------------------------
# Vectorizer and emotion model
# ---------------------------------------------------------------
vectorizer = CountVectorizer(
    token_pattern=r"[a-z]+(?:'[a-z]+)?|[?!]",
    stop_words=list(STOP),
    binary=True,
)
X_train = vectorizer.fit_transform(train_utt["utterance"])
X_val = vectorizer.transform(val_utt["utterance"])

emotion_model = MultinomialNB(alpha=0.5, fit_prior=False)
emotion_model.fit(X_train, train_utt["emotion"])

sentiment_model = MultinomialNB(alpha=0.5, fit_prior=False)
sentiment_model.fit(X_train, train_utt["sentiment"])

print("=== MELD: SENTIMENT (validation / dev) ===")
print(
    classification_report(
        val_utt["sentiment"], sentiment_model.predict(X_val), zero_division=0
    )
)

print("=== MELD: EMOTION (validation / dev) ===")
print(
    classification_report(
        val_utt["emotion"], emotion_model.predict(X_val), zero_division=0
    )
)

# ---------------------------------------------------------------
# Save
# ---------------------------------------------------------------
joblib.dump(
    {
        "vectorizer": vectorizer,
        "emotion_model": emotion_model,
        "sentiment_model": sentiment_model,
    },
    MODEL_PATH,
)
print("Saved to", MODEL_PATH)
