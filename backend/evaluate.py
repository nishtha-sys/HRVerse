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
from .classifier import build_model
from .test_data import TEST_DATA


def substring_intent(message):
    """The ORIGINAL first version: raw substring matching, no preprocessing."""
    text = message.lower()
    best_tag, best_score = None, 0
    for intent in bot.INTENTS:
        score = sum(1 for pattern in intent["patterns"] if pattern in text)
        if score > best_score:
            best_tag, best_score = intent["tag"], score
    return best_tag


def keyword_only(message):
    """Version 2: preprocessing (tokens, stop words, typo fix, stemming) + keyword patterns."""
    return bot.keyword_intent(bot.nlp.preprocess(message, bot.VOCAB))


def full_system(message):
    """Version 3 (current): ML classifier first, keyword patterns as safety net."""
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

    # Two different labels for questions that look identical after preprocessing confuse the model.
    seen = {}
    for text, label in bot.TRAINING_DATA:
        seen.setdefault(bot.CLASSIFIER.prepare(text), set()).add(label)
    conflicts = {k: v for k, v in seen.items() if len(v) > 1}
    print(f"  conflicting training examples: {len(conflicts)}")
    for prepared_text, group in conflicts.items():
        print(f"    '{prepared_text}' is labelled {sorted(group)}")

    print("\n" + "=" * 64)
    print("1. CROSS-VALIDATION on training data (5 folds, model only)")
    prepared = [bot.CLASSIFIER.prepare(t) for t in texts]
    scores = cross_val_score(build_model(), prepared, labels,
                             cv=StratifiedKFold(5, shuffle=True, random_state=42))
    print(f"  accuracy per fold : {[round(float(s), 3) for s in scores]}")
    print(f"  mean accuracy     : {scores.mean():.1%}")

    # Choose the confidence threshold fairly: from cross-validation on the TRAINING data only.
    folds = StratifiedKFold(5, shuffle=True, random_state=42)
    probabilities = cross_val_predict(build_model(), prepared, labels, cv=folds, method="predict_proba")
    classes = build_model().fit(prepared, labels).classes_
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
    print("4. CONFIDENCE THRESHOLD (how sure the model must be before it answers)")
    original = bot.CONFIDENCE_THRESHOLD
    for threshold in (0.2, 0.3, 0.4, 0.5, 0.6, 0.7):
        bot.CONFIDENCE_THRESHOLD = threshold
        marker = "  <- current" if threshold == original else ""
        print(f"  threshold {threshold:.1f}: accuracy {accuracy(full_system, TEST_DATA):6.1%}{marker}")
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