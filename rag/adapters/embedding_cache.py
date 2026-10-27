"""A vector cache in front of any embedder.

Re-indexing used to embed every chunk again (~7-10 min on CPU), even when a code change left
almost all chunk texts identical. The cache remembers each vector under a key of
(model name, exact chunk text), so only new or edited chunks reach the model. Changing the
embedding model changes every key, so a model switch can never reuse stale vectors.

It lives in its own file (data/cache/), not in the vector index, so deleting the index for a
clean rebuild does not throw the cache away.
"""
from __future__ import annotations

import hashlib
import sqlite3
from array import array
from pathlib import Path

from ..ports import Embedder


class CachingEmbedder:
    def __init__(self, inner: Embedder, path: str | Path, model_id: str):
        self.inner = inner
        self.model_id = model_id
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        # Streamlit caches one embedder and calls it from different script threads.
        self._db = sqlite3.connect(str(path), check_same_thread=False)
        self._db.execute("CREATE TABLE IF NOT EXISTS vectors (key TEXT PRIMARY KEY, vec BLOB)")
        self.hits = 0
        self.misses = 0

    def _key(self, text: str) -> str:
        return hashlib.sha256(f"{self.model_id}\n{text}".encode("utf-8")).hexdigest()

    def _get(self, keys: list[str]) -> dict[str, list[float]]:
        found: dict[str, list[float]] = {}
        for start in range(0, len(keys), 500):  # SQLite caps the number of ? parameters
            part = keys[start:start + 500]
            rows = self._db.execute(
                f"SELECT key, vec FROM vectors WHERE key IN ({','.join('?' * len(part))})", part)
            for key, blob in rows:
                found[key] = array("f", blob).tolist()
        return found

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        keys = [self._key(t) for t in texts]
        cached = self._get(list(dict.fromkeys(keys)))
        missing = list(dict.fromkeys(k for k in keys if k not in cached))
        self.hits += sum(1 for k in keys if k in cached)
        self.misses += len(missing)
        if missing:
            text_of = dict(zip(keys, texts))
            # One call for all misses: sentence-transformers sorts a call's inputs by length,
            # so batches hold similar-length chunks and waste less padding (~20% faster than
            # calls of 64 measured on this corpus).
            fresh = self.inner.embed_documents([text_of[k] for k in missing])
            with self._db:
                self._db.executemany("INSERT OR REPLACE INTO vectors VALUES (?, ?)",
                                     [(k, array("f", v).tobytes()) for k, v in zip(missing, fresh)])
            cached.update(zip(missing, fresh))
        return [cached[k] for k in keys]

    def embed_query(self, text: str) -> list[float]:
        # Not cached: queries are one-off and cheap, and bge adds a query-only prefix.
        return self.inner.embed_query(text)

    def close(self) -> None:
        self._db.close()
