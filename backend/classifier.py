"""
HRVerse - Intent Classification (Chapter 3.3, algorithm 3).

How it works
    1. Every question is turned into two sets of features:
         keywords  stop words removed, words reduced to their root ("leaves" -> "leav")
         phrases   the full wording with small words kept, so "what can you do" still means something
    2. TF-IDF turns both feature sets into numbers.
    3. Logistic Regression, trained on training_data.py, picks the most likely intent
       and gives a confidence between 0 and 1.

The model is trained when the server starts (well under a second), so there is no
model file to keep in sync: edit training_data.py, restart, done.
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline


def _ngram_analyzer(tokens_of, longest):
    """Features = single words plus neighbouring word groups up to `longest` words."""
    def analyze(text):
        tokens = tokens_of(text)
        features = list(tokens)
        for n in range(2, longest + 1):
            features.extend(" ".join(tokens[i:i + n]) for i in range(len(tokens) - n + 1))
        return features
    return analyze


class IntentClassifier:
    def __init__(self, keyword_tokens, phrase_tokens):
        """Both arguments are functions that turn a question into a list of tokens."""
        self.keyword_tokens = keyword_tokens
        self.phrase_tokens = phrase_tokens
        self.model = self.new_model()

    def new_model(self):
        """A fresh, untrained model (evaluate.py uses this for cross-validation)."""
        features = FeatureUnion([
            ("keywords", TfidfVectorizer(analyzer=_ngram_analyzer(self.keyword_tokens, 2), sublinear_tf=True)),
            ("phrases", TfidfVectorizer(analyzer=_ngram_analyzer(self.phrase_tokens, 3), sublinear_tf=True)),
        ])
        return Pipeline([
            ("features", features),
            ("clf", LogisticRegression(C=3, max_iter=2000, class_weight="balanced")),
        ])

    def fit(self, examples):
        """`examples` is a list of (question, intent) pairs."""
        self.model.fit([text for text, _ in examples], [label for _, label in examples])
        return self

    def rank(self, text):
        """All intents with their probability, most likely first. Empty if nothing usable is left."""
        if not self.phrase_tokens(text):
            return []
        probabilities = self.model.predict_proba([text])[0]
        pairs = zip(self.model.classes_, probabilities)
        return sorted(((str(tag), float(p)) for tag, p in pairs), key=lambda pair: -pair[1])

    def predict(self, text):
        """Return (intent, confidence), or (None, 0.0) if the message has no usable words."""
        ranked = self.rank(text)
        return ranked[0] if ranked else (None, 0.0)