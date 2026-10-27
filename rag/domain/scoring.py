"""Retrieval scoring against the golden set. Pure functions.

A question is a HIT if any chunk in the top k is on one of its expected pages. A question can
list several expected pages because the same rule is often stated in more than one place
(e.g. 42 CFR 422.204 and the Chapter 6 manual both say "recredential every 3 years").

MRR (mean reciprocal rank) adds how HIGH the first right chunk landed: rank 1 scores 1.0,
rank 2 scores 0.5, rank 5 scores 0.2, a miss scores 0. It separates "barely found" from
"found first", which hit@k alone cannot.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from .models import Chunk


@dataclass(frozen=True)
class Expected:
    doc: str  # prefix of the chunk's doc_id, e.g. "ecfr_42cfr_422.204"
    page: int
    # A phrase from the answering passage that the CHUNK must contain ("a | b" = either).
    # Needed since chunks follow headings: Chapter 6 page 4 holds both section 20.2 (the
    # consultation answer) and section 30 (written notice), and a page-only match counted the
    # section 30 chunk as a hit. Empty = any chunk on the page counts.
    contains: str = ""


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.replace("’", "'")).strip().lower()


def is_relevant(chunk: Chunk, expected: list[Expected]) -> bool:
    for e in expected:
        if not (chunk.doc_id.startswith(e.doc) and chunk.page == e.page):
            continue
        if not e.contains or any(_norm(alt) in _norm(chunk.text) for alt in e.contains.split("|")):
            return True
    return False


def first_relevant_rank(chunks: list[Chunk], expected: list[Expected]) -> int | None:
    for rank, chunk in enumerate(chunks, start=1):
        if is_relevant(chunk, expected):
            return rank
    return None


@dataclass(frozen=True)
class Summary:
    questions: int
    hits: int
    mrr: float

    @property
    def hit_rate(self) -> float:
        return self.hits / self.questions if self.questions else 0.0


def summarise(ranks: list[int | None], k: int) -> Summary:
    """ranks: first relevant rank per question (None = not retrieved at all)."""
    in_top_k = [r for r in ranks if r is not None and r <= k]
    mrr = sum(1.0 / r for r in in_top_k) / len(ranks) if ranks else 0.0
    return Summary(questions=len(ranks), hits=len(in_top_k), mrr=mrr)
