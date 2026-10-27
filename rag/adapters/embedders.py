"""Embedders: a free local sentence-transformers model, and a hashing fake for tests."""
from __future__ import annotations

import hashlib
import math
import re

DEFAULT_EMBED_MODEL = "BAAI/bge-small-en-v1.5"
# bge models retrieve better when the QUERY (not the passage) carries this instruction.
_BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class SentenceTransformerEmbedder:
    """Runs on CPU, nothing leaves the machine. The model (~130MB for bge-small) downloads
    once on first use into the Hugging Face cache."""

    def __init__(self, model_name: str = DEFAULT_EMBED_MODEL, batch_size: int = 32):
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.batch_size = batch_size
        self._model = SentenceTransformer(model_name)
        self._query_prefix = _BGE_QUERY_PREFIX if "bge" in model_name.lower() else ""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors = self._model.encode(texts, batch_size=self.batch_size,
                                     normalize_embeddings=True, show_progress_bar=False)
        return [v.tolist() for v in vectors]

    def embed_query(self, text: str) -> list[float]:
        vector = self._model.encode(self._query_prefix + text, normalize_embeddings=True)
        return vector.tolist()


_TOKEN = re.compile(r"[a-z0-9]+")


class HashingEmbedder:
    """Deterministic bag-of-words vectors: each word is hashed into one of `dim` buckets.
    Not semantic at all, but texts that share words score higher, which is enough to test
    that retrieval, ranking and citations are wired correctly - at zero cost."""

    def __init__(self, dim: int = 256):
        self.dim = dim

    def _vector(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for word in _TOKEN.findall(text.lower()):
            bucket = int(hashlib.md5(word.encode()).hexdigest(), 16) % self.dim
            vec[bucket] += 1.0
        norm = math.sqrt(sum(x * x for x in vec)) or 1.0
        return [x / norm for x in vec]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)
