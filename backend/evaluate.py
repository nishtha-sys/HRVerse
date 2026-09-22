"""
HRVerse - evaluation (Chapter 5 "Results and Discussion").

Run from the project root:
    python -m backend.evaluate

Prints numbers you can copy into the report: accuracy, per-intent precision/recall,
a comparison of the three matching methods, and the average response time.
"""

import time
from collections import Counter

from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score

from . import main as bot
from .test_data import TEST_DATA

# The earlier versions only knew these topics. The newer skills (about the bot, what it can do,
# acting on someone's behalf, private information) did not exist yet, so they count as misses.
ORIGINAL_TAGS = {"greeting", "thanks", "apply_leave", "leave_policy", "salary", "wfh", "working_hours",
                 "probation", "contact_hr", "benefits", "ethics", "legal", "bad_language"}


def substring_intent(message):
    """Version 1: raw substring matching, no preprocessing."""
    text = message.lower()
    best_tag, best_score = None, 0
    for intent in bot.INTENTS:
        if intent["tag"] not in ORIGINAL_TAGS:
            continue
        score = sum(1 for pattern in intent["patterns"] if pattern in text)
        if score > best_score:
            best_tag, best_score = intent["tag"], score
    return best_tag


def keyword_only(message):
    """Version 2: preprocessing (tokens, stop words, typo fix, stemming) + keyword patterns."""
    tokens = bot.nlp.preprocess(message, bot.VOCAB)
    best_tag, best_score = None, 0
    for tag, patterns in bot.PATTERN_TOKENS:
        if tag not in ORIGINAL_TAGS:
            continue
        score = sum(1 for pattern in patterns if bot.nlp.contains_phrase(tokens, pattern))
        if score > best_score:
            best_tag, best_score = tag, score
    return best_tag


def full_system(message):
    """Version 3 (current): ML classifier first, keyword patterns and the privacy rule as safety nets."""
    if bot.asks_about_others_privately(bot.nlp.normalize(message)):
        return "personal_info"
    return bot.predict_intent(message)[0]


def accuracy(method, data):
    """A question the bot declines to answer (None) counts as 'out_of_scope'."""
    hits = 0
    for text, label in data:
        predicted = method(text) or "out_of_scope"
        hits += predicted == label
    return hits / len(data)


def main():
    texts = [t for t, _ in bot.TRAINING_DATA]
    labels = [l for _, l in bot.TRAINING_DATA]
    overlap = {t for t, _ in TEST_DATA} & set(texts)

    print("=" * 64)
    print("DATA")
    print(f"  training questions : {len(bot.TRAINING_DATA)}  ({len(set(labels))} intents)")
    print(f"  test questions     : {len(TEST_DATA)}  (never used for training)")
    print(f"  test questions also in training data: {len(overlap)}")

    # The same wording with two different labels confuses the model.
    seen = {}
    for text, label in bot.TRAINING_DATA:
        seen.setdefault(" ".join(bot.nlp.normalize(text, bot.VOCAB)), set()).add(label)
    conflicts = {k: v for k, v in seen.items() if len(v) > 1}
    print(f"  conflicting training examples: {len(conflicts)}")
    for wording, group in conflicts.items():
        print(f"    '{wording}' is labelled {sorted(group)}")

    print("\n" + "=" * 64)
    print("1. CROSS-VALIDATION on training data (5 folds, model only)")
    folds = StratifiedKFold(5, shuffle=True, random_state=42)
    scores = cross_val_score(bot.CLASSIFIER.new_model(), texts, labels, cv=folds)
    print(f"  accuracy per fold : {[round(float(s), 3) for s in scores]}")
    print(f"  mean accuracy     : {scores.mean():.1%}")

    # Choose the confidence threshold fairly: from cross-validation on the TRAINING data only.
    probabilities = cross_val_predict(bot.CLASSIFIER.new_model(), texts, labels, cv=folds, method="predict_proba")
    classes = bot.CLASSIFIER.new_model().fit(texts, labels).classes_
    print("  threshold vs cross-validated accuracy (model answers only if it is at least this sure):")
    for threshold in (0.2, 0.3, 0.4, 0.5, 0.6):
        hits = 0
        for row, label in zip(probabilities, labels):
            guess = classes[row.argmax()] if row.max() >= threshold else "out_of_scope"
            hits += guess == label
        marker = "  <- current" if threshold == bot.CONFIDENCE_THRESHOLD else ""
        print(f"    {threshold:.1f}: {hits / len(labels):6.1%}{marker}")

    print("\n" + "=" * 64)
    print("2. HELD-OUT TEST SET: the three matching methods compared")
    rows = [
        ("v1  raw substring matching (original)", substring_intent),
        ("v2  + NLP preprocessing, keyword patterns", keyword_only),
        ("v3  + ML classifier (current system)", full_system),
    ]
    for name, method in rows:
        print(f"  {name:45s} {accuracy(method, TEST_DATA):6.1%}")

    print("\n" + "=" * 64)
    print("3. CURRENT SYSTEM, per intent (held-out test set)")
    expected = [label for _, label in TEST_DATA]
    predicted = [(full_system(text) or "out_of_scope") for text, _ in TEST_DATA]
    print(classification_report(expected, predicted, zero_division=0, digits=2))

    wrong = [(text, label, pred) for (text, label), pred in zip(TEST_DATA, predicted) if label != pred]
    print(f"Mistakes ({len(wrong)} of {len(TEST_DATA)}):")
    for text, label, pred in wrong:
        print(f"  {text!r}: expected {label}, got {pred}")
    if not wrong:
        print("  none")

    print("\n" + "=" * 64)
    print("4. CONFIDENCE THRESHOLD on the test set")
    original = bot.CONFIDENCE_THRESHOLD
    for threshold in (0.2, 0.3, 0.35, 0.4, 0.5, 0.6):
        bot.CONFIDENCE_THRESHOLD = threshold
        marker = "  <- current" if threshold == original else ""
        print(f"  threshold {threshold:.2f}: accuracy {accuracy(full_system, TEST_DATA):6.1%}{marker}")
    bot.CONFIDENCE_THRESHOLD = original

    print("\n" + "=" * 64)
    print("5. SPEED (report target: 1-2 seconds per reply)")
    start = time.perf_counter()
    rounds = 20
    for _ in range(rounds):
        for text, _ in TEST_DATA:
            bot.get_response(text)
    per_message = (time.perf_counter() - start) / (rounds * len(TEST_DATA)) * 1000
    print(f"  average time per message: {per_message:.2f} ms")

    counts = Counter(method for _, _, method in (bot.predict_intent(t) for t, _ in TEST_DATA))
    print(f"\n  answered by: {dict(counts)}")


if __name__ == "__main__":
    main()