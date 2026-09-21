"""
HRVerse - Intent Classification (Chapter 3.3, algorithm 3).

How it works
    1. Every question goes through the preprocessing pipeline (preprocess.py).
    2. TF-IDF turns the remaining words (and word pairs) into numbers.
    3. Logistic Regression, trained on training_data.py, picks the most likely intent
       and gives a confidence between 0 and 1.

The model is trained when the server starts (well under a second), so there is no
model file to keep in sync: edit training_data.py, restart, done.
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


def build_model():
    return Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)),
        ("clf", LogisticRegression(C=20, max_iter=2000, class_weight="balanced")),
    ])


class IntentClassifier:
    def __init__(self, prepare):
        """`prepare(text)` must return the preprocessed text as one string."""
        self.prepare = prepare
        self.model = build_model()

    def fit(self, examples):
        """`examples` is a list of (question, intent) pairs."""
        texts = [self.prepare(text) for text, _ in examples]
        labels = [label for _, label in examples]
        self.model.fit(texts, labels)
        return self

    def predict(self, text):
        """Return (intent, confidence). Returns (None, 0.0) if nothing usable is left after preprocessing."""
        prepared = self.prepare(text)
        if not prepared.strip():
            return None, 0.0
        probabilities = self.model.predict_proba([prepared])[0]
        best = probabilities.argmax()
        return self.model.classes_[best], float(probabilities[best])