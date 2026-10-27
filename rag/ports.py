"""The seams between the pipeline and the outside world. Each one has a real adapter and a
fake, so tests never need a model download, a network call or an API key."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from .domain.models import Chunk, Hit, Page


class Loader(Protocol):
    def load(self, path: Path) -> list[Page]: ...


class Embedder(Protocol):
    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class VectorStore(Protocol):
    def upsert(self, chunks: list[Chunk], vectors: list[list[float]]) -> None: ...

    def delete_doc(self, doc_id: str) -> None: ...

    def search(self, vector: list[float], k: int) -> list[Hit]: ...

    def count(self) -> int: ...

    def sources(self) -> list[str]: ...

    def all_chunks(self) -> list[Chunk]: ...


class Reranker(Protocol):
    def score(self, question: str, texts: list[str]) -> list[float]:
        """One relevance score per text, higher is more relevant. Only the order matters."""
        ...


@dataclass(frozen=True)
class LLMResult:
    text: str
    stop_reason: str = "end_turn"
    input_tokens: int = 0
    output_tokens: int = 0
    model: str = ""


class LLM(Protocol):
    def complete(self, *, system: str, user: str, max_tokens: int = 1024,
                 json_schema: dict | None = None) -> LLMResult:
        """json_schema, when given, asks for a JSON reply matching it (the judge uses this)."""
        ...
