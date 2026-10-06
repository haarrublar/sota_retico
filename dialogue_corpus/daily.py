import joblib
import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, CountVectorizer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

# ---------------------------------------------------------------
# Settings
# ---------------------------------------------------------------
DATA_PATH = "DAILYDIALOG/data/dialogues.json"
GAME_PATH = "DAILYDIALOG/game_sentences.csv"
MODEL_PATH = "DAILYDIALOG/dailydialog_models.joblib"
DOMAIN = "Relationship"  # None = all domains
GAME_REPEAT = 10  # how many times the game sentences are added to training

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
# Load DailyDialog and flatten it to one row per utterance
# ---------------------------------------------------------------
all_dialogues = pd.read_json(DATA_PATH)


def select_split(split):
    db = all_dialogues[all_dialogues["data_split"] == split]
    if DOMAIN is not None:
        db = db[db["domains"].apply(lambda d: DOMAIN in d)]
    return db


def make_utterances(dialogues):
    rows = []
    for _, r in dialogues.iterrows():
        for t in r["turns"]:
            acts = t["dialogue_acts"]["binary"]
            rows.append(
                {
                    "utterance": t["utterance"],
                    "intent": acts[0]["intent"] if acts else None,
                    "emotion": t["emotion"],
                }
            )
    return pd.DataFrame(rows).dropna(subset=["intent"])


dd_train = make_utterances(select_split("train"))
dd_val = make_utterances(select_split("validation"))

# ---------------------------------------------------------------
# Load the game sentences and split them 80% train / 20% validation
# ---------------------------------------------------------------
rows = []
with open(GAME_PATH, encoding="utf-8") as f:
    next(f)  # skip the header line
    for line in f:
        line = line.strip()
        if not line:  # skip empty lines
            continue
        utterance, intent = line.rsplit(",", 1)  # split only at the last comma
        rows.append({"utterance": utterance.strip().strip('"'), "intent": intent})

game = pd.DataFrame(rows)
game["intent"] = game["intent"].str.strip().str.lower()  # "Inform " -> "inform"

game_train, game_val = train_test_split(
    game,
    test_size=0.2,
    stratify=game["intent"],  # same intent mix in both parts
    random_state=42,  # same split every run
)

# ---------------------------------------------------------------
# Mix: DailyDialog + game sentences (repeated to give them weight)
# ---------------------------------------------------------------
intent_train = pd.concat(
    [dd_train[["utterance", "intent"]]] + [game_train] * GAME_REPEAT,
    ignore_index=True,
)

print("DailyDialog train utterances:", len(dd_train))
print("Game train sentences:", len(game_train), f"(x{GAME_REPEAT})")
print("Game validation sentences:", len(game_val))

# ---------------------------------------------------------------
# One vectorizer, learned on the mixed text (includes game words)
# ---------------------------------------------------------------
vectorizer = CountVectorizer(
    token_pattern=r"[a-z]+(?:'[a-z]+)?|[?!]",
    stop_words=list(STOP),
    binary=True,
)
X_intent_train = vectorizer.fit_transform(intent_train["utterance"])

# ---------------------------------------------------------------
# Intent model: trained on the mix, checked on BOTH validation sets
# ---------------------------------------------------------------
intent_model = MultinomialNB(alpha=0.5, fit_prior=False)
intent_model.fit(X_intent_train, intent_train["intent"])

print("\n=== INTENT on DailyDialog validation ===")
pred = intent_model.predict(vectorizer.transform(dd_val["utterance"]))
print(classification_report(dd_val["intent"], pred, zero_division=0))

print("=== INTENT on game validation ===")
pred = intent_model.predict(vectorizer.transform(game_val["utterance"]))
print(classification_report(game_val["intent"], pred, zero_division=0))

# ---------------------------------------------------------------
# Emotion model: DailyDialog only (game sentences have no emotion)
# ---------------------------------------------------------------
emotion_model = MultinomialNB(alpha=0.5, fit_prior=False)
emotion_model.fit(vectorizer.transform(dd_train["utterance"]), dd_train["emotion"])

print("=== EMOTION on DailyDialog validation ===")
pred = emotion_model.predict(vectorizer.transform(dd_val["utterance"]))
print(classification_report(dd_val["emotion"], pred, zero_division=0))

# ---------------------------------------------------------------
# Save
# ---------------------------------------------------------------
joblib.dump(
    {
        "vectorizer": vectorizer,
        "intent_model": intent_model,
        "emotion_model": emotion_model,
    },
    MODEL_PATH,
)
print("Saved to", MODEL_PATH)
