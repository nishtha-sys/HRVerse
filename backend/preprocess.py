"""
HRVerse - Input Processing Layer (Chapter 3.3 of the project report).

Pipeline for every user message:
    1. Text normalisation   lowercase, unify apostrophes, drop punctuation/emojis
    2. Tokenisation         split the text into words (NLTK RegexpTokenizer)
    3. Stop-word removal    drop words that carry no intent ("the", "is", "can" ...)
    4. Spelling correction  fix typos using words the bot already knows ("probaton" -> "probation")
    5. Stemming             reduce words to a root (NLTK SnowballStemmer): "leaves" -> "leav"

Nothing here needs a download: NLTK's tokenizer and stemmer are pure code.
"""

import difflib

from nltk.stem import SnowballStemmer
from nltk.tokenize import RegexpTokenizer

_tokenizer = RegexpTokenizer(r"[a-z0-9]+")     # letters and digits only
_stemmer = SnowballStemmer("english")

# Words that never decide the intent of an HR question.
# Quantity words such as "many" and "much" are kept on purpose.
STOP_WORDS = {
    "a", "an", "the", "and", "or", "but", "if", "so",
    "is", "are", "am", "was", "were", "be", "been", "being",
    "do", "does", "did", "have", "has", "had",
    "i", "me", "my", "mine", "we", "us", "our", "you", "your", "he", "she", "it", "its", "they", "them", "their",
    "to", "of", "in", "on", "at", "for", "from", "with", "by", "about", "into",
    "can", "could", "would", "should", "will", "shall", "may", "might", "must",
    "what", "whats", "how", "when", "where", "which", "who", "why",
    "there", "this", "that", "these", "those",
    "please", "plz", "pls", "kindly", "tell", "know",
    "s", "t", "m", "re", "ve", "ll", "d",
}

MIN_LENGTH_TO_CORRECT = 4       # very short words are too ambiguous to auto-correct
CORRECTION_CUTOFF = 0.90        # how similar a known word must be (0 to 1); 'play' must NOT become 'pay'


def tokenize(text):
    """Normalise and split into lowercase word tokens (no stop-word removal yet)."""
    text = text.lower().replace("\u2019", "'").replace("\u2018", "'")
    return _tokenizer.tokenize(text)


def stem(word):
    return _stemmer.stem(word)


def build_vocabulary(phrases):
    """Collect the known (non stop-word) words from a list of phrases."""
    vocab = set()
    for phrase in phrases:
        for token in tokenize(phrase):
            if token not in STOP_WORDS:
                vocab.add(token)
    return vocab


def correct(token, vocab):
    """Return the closest known word if `token` looks like a typo, otherwise the token itself."""
    if token in vocab or len(token) < MIN_LENGTH_TO_CORRECT or token.isdigit():
        return token
    close = difflib.get_close_matches(token, vocab, n=1, cutoff=CORRECTION_CUTOFF)
    return close[0] if close else token


def preprocess(text, vocab=None):
    """Full pipeline. Returns the list of stemmed tokens for a message or a pattern."""
    tokens = [t for t in tokenize(text) if t not in STOP_WORDS]
    if vocab:
        tokens = [correct(t, vocab) for t in tokens]
    return [stem(t) for t in tokens]


def contains_phrase(tokens, phrase_tokens):
    """True if `phrase_tokens` appear next to each other, in order, inside `tokens`."""
    n = len(phrase_tokens)
    if n == 0 or n > len(tokens):
        return False
    return any(tokens[i:i + n] == phrase_tokens for i in range(len(tokens) - n + 1))