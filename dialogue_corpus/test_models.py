import time

import joblib
import pandas as pd

# ---------------------------------------------------------------
# Load the saved models (no data, no training)
# ---------------------------------------------------------------
dd = joblib.load("DAILYDIALOG/dailydialog_models.joblib")
meld = joblib.load("MELD/meld_models.joblib")


def fix_text(text):
    # same cleanup used when training on MELD
    for bad in ["\u2019", "\u2018", "\u00e2\u0080\u0099", "\x92", "\u00e2\u20ac\u2122"]:
        text = text.replace(bad, "'")
    return text


def probs(bundle, model_name, text):
    X = bundle["vectorizer"].transform([fix_text(text)])
    model = bundle[model_name]
    return pd.Series(model.predict_proba(X)[0], index=model.classes_)


# ---------------------------------------------------------------
# Try some sentences
# ---------------------------------------------------------------
sentences = [
    "Will you betray me?",
    "I promise I will cooperate this time.",
    "You betrayed me!",
    "Let's both cooperate.",
    "I'm so happy we won together!",
]

for s in sentences:
    start = time.perf_counter()
    intent = probs(dd, "intent_model", s)
    emotion_dd = probs(dd, "emotion_model", s)
    emotion_meld = probs(meld, "emotion_model", s)
    elapsed = time.perf_counter() - start

    print(f"\n{s}")
    print(f"  intent (DailyDialog):   {intent.idxmax():12s} {intent.max():.2f}")
    print(f"  emotion (DailyDialog):  {emotion_dd.idxmax():12s} {emotion_dd.max():.2f}")
    print(
        f"  emotion (MELD):         {emotion_meld.idxmax():12s} {emotion_meld.max():.2f}"
    )
    print(f"  time for all three:     {elapsed * 1000:.2f} ms")

# Full probabilities for one sentence
print("\nAll probabilities for:", sentences[0])
print(pd.DataFrame({"intent": probs(dd, "intent_model", sentences[0])}).round(3))
print(
    pd.DataFrame(
        {
            "DailyDialog": probs(dd, "emotion_model", sentences[0]),
            "MELD": probs(meld, "emotion_model", sentences[0]),
        }
    ).round(3)
)
