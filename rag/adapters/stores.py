"""Vector stores: Chroma persisted to a local folder, and an in-memory one for tests."""
from __future__ import annotations

import math
from pathlib import Path

from ..domain.models import Chunk, Hit

COLLECTION = "credentialing"


def _chunk_from(chunk_id: str, text: str, meta: dict) -> Chunk:
    return Chunk(id=chunk_id, doc_id=meta["doc_id"], source=meta["source"],
                 page=int(meta["page"]), text=text)


class ChromaStore:
    def __init__(self, path: str | Path, collection: str = COLLECTION):
        import chromadb

        Path(path).mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(path))
        # Cosine, because the embedders return normalised vectors; Chroma's default is L2.
        self._col = self._client.get_or_create_collection(
            collection, metadata={"hnsw:space": "cosine"})

    def upsert(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        if not chunks:
            return
        self._col.upsert(
            ids=[c.id for c in chunks],
            embeddings=vectors,
            documents=[c.text for c in chunks],
            metadatas=[{"doc_id": c.doc_id, "source": c.source, "page": c.page}
                       for c in chunks],
        )

    def delete_doc(self, doc_id: str) -> None:
        self._col.delete(where={"doc_id": doc_id})

    def search(self, vector: list[float], k: int) -> list[Hit]:
        if self.count() == 0:
            return []
        res = self._col.query(query_embeddings=[vector], n_results=min(k, self.count()),
                              include=["documents", "metadatas", "distances"])
        return [Hit(chunk=_chunk_from(cid, doc, meta), score=1.0 - dist)
                for cid, doc, meta, dist in zip(res["ids"][0], res["documents"][0],
                                                res["metadatas"][0], res["distances"][0])]

    def count(self) -> int:
        return self._col.count()

    def sources(self) -> list[str]:
        metas = self._col.get(include=["metadatas"])["metadatas"] or []
        return sorted({m["source"] for m in metas})

    def all_chunks(self) -> list[Chunk]:
        res = self._col.get(include=["documents", "metadatas"])
        return [_chunk_from(cid, doc, meta)
                for cid, doc, meta in zip(res["ids"], res["documents"], res["metadatas"])]


class MemoryStore:
    def __init__(self):
        self._rows: dict[str, tuple[Chunk, list[float]]] = {}

    def upsert(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        for chunk, vec in zip(chunks, vectors, strict=True):
            self._rows[chunk.id] = (chunk, vec)

    def delete_doc(self, doc_id: str) -> None:
        self._rows = {k: v for k, v in self._rows.items() if v[0].doc_id != doc_id}

    def search(self, vector: list[float], k: int) -> list[Hit]:
        def cosine(a, b):
            dot = sum(x * y for x, y in zip(a, b))
            na = math.sqrt(sum(x * x for x in a)) or 1.0
            nb = math.sqrt(sum(y * y for y in b)) or 1.0
            return dot / (na * nb)

        hits = [Hit(chunk=c, score=cosine(vector, v)) for c, v in self._rows.values()]
        hits.sort(key=lambda h: (-h.score, h.chunk.id))
        return hits[:k]

    def count(self) -> int:
        return len(self._rows)

    def sources(self) -> list[str]:
        return sorted({c.source for c, _ in self._rows.values()})

    def all_chunks(self) -> list[Chunk]:
        return [c for c, _ in self._rows.values()]
