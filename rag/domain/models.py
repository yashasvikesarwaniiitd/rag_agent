"""Plain data that flows through the pipeline. No behaviour, no SDK types."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Line:
    text: str
    heading: bool = False


@dataclass(frozen=True)
class Page:
    """One page of extracted text. `page` is 1-based, matching what a reader sees in a PDF
    viewer, because that is the number a citation has to point at.

    `lines` is the same text with heading lines marked, when the loader can tell (font size,
    bold, HTML tags). Empty means "no structure known" and the chunker falls back to `text`.
    """
    doc_id: str
    source: str
    page: int
    text: str
    lines: tuple[Line, ...] = ()
    doc_title: str = ""


@dataclass(frozen=True)
class Chunk:
    id: str
    doc_id: str
    source: str
    page: int
    text: str
    # Where the chunk sits: "document title > section > subsection". Shown to the model and
    # in citations, but deliberately NOT indexed: measured on the golden set, putting it in
    # the index cost 2/20 questions, because a section title like "Credentialing, Monitoring,
    # and Recredentialing" is stamped on ~20 chunks and drowns the one rare word (BM25) or
    # blurs the vectors. See evals/results/v0.1-structured.md.
    context: str = ""


@dataclass(frozen=True)
class Hit:
    chunk: Chunk
    score: float  # cosine similarity (vector) or fused rank score (hybrid); higher is better


@dataclass(frozen=True)
class Citation:
    n: int  # the [n] marker used in the answer text
    source: str
    page: int
    chunk_id: str
    snippet: str
    context: str = ""


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
    rewrites: list[str] = field(default_factory=list)  # extra search phrasings used, if any
    latency_ms: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    model: str = ""
