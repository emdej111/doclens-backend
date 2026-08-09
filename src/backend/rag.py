"""Simple chunk-based retrieval for RAG.

No external vector DB and no embedding API call: chunks are scored against a
question with a TF-IDF cosine-similarity index built entirely with numpy.
Good enough for single-document, in-memory retrieval at demo scale.
"""

import re
from dataclasses import dataclass, field

import numpy as np

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Split text into overlapping character-based chunks.

    Character-based (not token-based) chunking keeps this dependency-free;
    `chunk_size` characters is a reasonable proxy for a few hundred tokens.
    """
    text = text.strip()
    if not text:
        return []
    if overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    chunks = []
    start = 0
    step = chunk_size - overlap
    while start < len(text):
        chunk = text[start : start + chunk_size].strip()
        if chunk:
            chunks.append(chunk)
        start += step
    return chunks


@dataclass
class TfidfIndex:
    """A TF-IDF index over a fixed set of chunks, queryable by cosine similarity."""

    chunks: list[str]
    vocabulary: dict[str, int] = field(default_factory=dict)
    idf: np.ndarray = field(default_factory=lambda: np.empty(0))
    doc_vectors: np.ndarray = field(default_factory=lambda: np.empty((0, 0)))

    @classmethod
    def build(cls, chunks: list[str]) -> "TfidfIndex":
        tokenized = [_tokenize(chunk) for chunk in chunks]

        vocabulary: dict[str, int] = {}
        for tokens in tokenized:
            for token in tokens:
                vocabulary.setdefault(token, len(vocabulary))

        n_docs = len(chunks)
        n_terms = len(vocabulary)
        term_counts = np.zeros((n_docs, n_terms), dtype=np.float64)

        for row, tokens in enumerate(tokenized):
            for token in tokens:
                term_counts[row, vocabulary[token]] += 1.0

        doc_freq = np.count_nonzero(term_counts, axis=0)
        idf = np.log((n_docs + 1) / (doc_freq + 1)) + 1.0  # smoothed idf

        tfidf = term_counts * idf
        norms = np.linalg.norm(tfidf, axis=1, keepdims=True)
        norms[norms == 0] = 1.0  # avoid divide-by-zero for empty chunks
        doc_vectors = tfidf / norms

        return cls(chunks=chunks, vocabulary=vocabulary, idf=idf, doc_vectors=doc_vectors)

    def _vectorize_query(self, question: str) -> np.ndarray:
        vec = np.zeros(len(self.vocabulary), dtype=np.float64)
        for token in _tokenize(question):
            idx = self.vocabulary.get(token)
            if idx is not None:
                vec[idx] += 1.0
        vec *= self.idf
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return vec

    def query(self, question: str, top_k: int) -> list[tuple[int, float, str]]:
        """Return the top_k (chunk_index, score, chunk_text) results, best first."""
        if not self.chunks:
            return []

        query_vec = self._vectorize_query(question)
        scores = self.doc_vectors @ query_vec  # cosine similarity (vectors are unit-normalized)

        top_k = min(top_k, len(self.chunks))
        top_indices = np.argsort(scores)[::-1][:top_k]

        return [(int(i), float(scores[i]), self.chunks[i]) for i in top_indices]
