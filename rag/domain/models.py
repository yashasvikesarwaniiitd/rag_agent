"""Plain data that flows through the pipeline. No behaviour, no SDK types."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Page:
    """One page of extracted text. `page` is 1-based, matching what a reader sees in a PDF
    viewer, because that is the number a citation has to point at."""
    doc_id: str
    source: str
    page: int
    text: str


@dataclass(frozen=True)
class Chunk:
    id: str
    doc_id: str
    source: str
    page: int
    text: str


@dataclass(frozen=True)
class Hit:
    chunk: Chunk
    score: float  # cosine similarity, higher is closer


@dataclass(frozen=True)
class Citation:
    n: int  # the [n] marker used in the answer text
    source: str
    page: int
    chunk_id: str
    snippet: str


@dataclass
class Answer:
    text: str
    citations: list[Citation] = field(default_factory=list)
    hits: list[Hit] = field(default_factory=list)
    # True when we decline to answer: empty index, sources don't cover it, or the model's
    # answer failed the citation gate. `reason` says which, so it can be evaluated.
    refused: bool = False
    reason: str = ""
    raw_model_text: str = ""
    latency_ms: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    model: str = ""
