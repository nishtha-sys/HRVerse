"""
HRVerse - Policy Document Search ("Policy documents" card, roadmap item).

Lets an employee upload a policy PDF and ask questions answered from its actual text,
using retrieval (TF-IDF + cosine similarity) rather than a generative model, so it stays
consistent with the rest of this project's classical-NLP approach.

Storage is in memory only. Render's free plan wipes the disk on every restart, so
there is no reliable place to keep uploaded files between deploys without an external
database - documents need to be re-uploaded after the server restarts. This is stated
to the user in the UI, not hidden.
"""

import io

from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

CHUNK_WORDS = 180     # roughly a paragraph or two per chunk
CHUNK_OVERLAP = 40    # shared words between consecutive chunks, so an answer near a
                       # chunk boundary is not split in half
MIN_SCORE = 0.12       # below this similarity, treat it as "not found" rather than guess


def extract_text(pdf_bytes):
    """Read all the selectable text out of a PDF. Scanned/image-only PDFs return "" here."""
    reader = PdfReader(io.BytesIO(pdf_bytes))
    return "\n".join((page.extract_text() or "") for page in reader.pages)


def chunk_text(text, size=CHUNK_WORDS, overlap=CHUNK_OVERLAP):
    """Split into overlapping word-count chunks, so each is small enough to be a focused answer.
    Stops as soon as a chunk reaches the end of the text, so short documents (or the last part
    of a long one) don't get a tiny, redundant extra chunk that duplicates the previous one."""
    words = text.split()
    if not words:
        return []
    step = max(size - overlap, 1)
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + size, len(words))
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start += step
    return chunks


class PolicyIndex:
    """Holds every uploaded document's chunks and a TF-IDF index over all of them together."""

    def __init__(self):
        self.documents = []       # [{"name": ..., "chunk_count": ...}, ...] for display
        self.chunk_texts = []     # parallel arrays: one entry per chunk, across all documents
        self.chunk_source = []
        self.vectorizer = None
        self.matrix = None

    def add_document(self, name, text):
        chunks = chunk_text(text)
        self.documents.append({"name": name, "chunk_count": len(chunks)})
        self.chunk_texts.extend(chunks)
        self.chunk_source.extend([name] * len(chunks))
        self._rebuild_index()
        return len(chunks)

    def _rebuild_index(self):
        if not self.chunk_texts:
            self.vectorizer, self.matrix = None, None
            return
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
        self.matrix = self.vectorizer.fit_transform(self.chunk_texts)

    def has_documents(self):
        return bool(self.chunk_texts)

    def search(self, question, top_k=1):
        """Best-matching chunk(s) for `question`, above MIN_SCORE. Empty list if nothing is close enough."""
        if self.matrix is None:
            return []
        question_vector = self.vectorizer.transform([question])
        scores = cosine_similarity(question_vector, self.matrix)[0]
        ranked = scores.argsort()[::-1][:top_k]
        return [
            {"text": self.chunk_texts[i], "source": self.chunk_source[i], "score": float(scores[i])}
            for i in ranked if scores[i] >= MIN_SCORE
        ]


# One shared index for the whole running server (module-level singleton).
POLICY_INDEX = PolicyIndex()