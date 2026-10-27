"""Splitting pages into chunks. Pure functions.

v0 is deliberately simple: paragraphs are packed up to `max_chars`, with a character overlap
carried into the next chunk so a sentence cut at a boundary is still findable. Chunks never
cross a page boundary, which costs a little context but keeps every citation to ONE page.
Heading- and table-aware chunking is the v0.1 step.
"""
from __future__ import annotations

import re

from .models import Chunk, Page

_PARA_SPLIT = re.compile(r"\n\s*\n")
_WS = re.compile(r"[ \t]+")


def normalise(text: str) -> str:
    """Collapse runs of spaces and stray blank lines, keep paragraph breaks."""
    lines = [_WS.sub(" ", ln).strip() for ln in text.replace("\r", "").split("\n")]
    out = "\n".join(lines)
    return re.sub(r"\n{3,}", "\n\n", out).strip()


def _split_long(paragraph: str, max_chars: int) -> list[str]:
    """Hard-split a paragraph longer than max_chars, preferring sentence then word ends."""
    pieces: list[str] = []
    rest = paragraph
    while len(rest) > max_chars:
        window = rest[:max_chars]
        cut = max(window.rfind(". "), window.rfind("; "))
        if cut < max_chars // 2:
            cut = window.rfind(" ")
        if cut <= 0:
            cut = max_chars - 1
        pieces.append(rest[: cut + 1].strip())
        rest = rest[cut + 1:].strip()
    if rest:
        pieces.append(rest)
    return pieces


def _tail(text: str, overlap: int) -> str:
    """Last ~overlap chars, started on a word boundary so the overlap is readable."""
    if overlap <= 0 or len(text) <= overlap:
        return "" if overlap <= 0 else text
    tail = text[-overlap:]
    space = tail.find(" ")
    return tail[space + 1:] if 0 <= space < len(tail) - 1 else tail


def chunk_page(page: Page, max_chars: int = 1200, overlap: int = 150) -> list[Chunk]:
    if overlap >= max_chars:
        raise ValueError("overlap must be smaller than max_chars")
    text = normalise(page.text)
    if not text:
        return []

    paragraphs: list[str] = []
    for para in _PARA_SPLIT.split(text):
        para = para.strip()
        if para:
            paragraphs.extend(_split_long(para, max_chars - overlap))

    bodies: list[str] = []
    current = ""
    for para in paragraphs:
        candidate = f"{current}\n\n{para}" if current else para
        if len(candidate) <= max_chars:
            current = candidate
            continue
        bodies.append(current)
        carry = _tail(current, overlap)
        current = f"{carry}\n\n{para}" if carry else para
    if current:
        bodies.append(current)

    return [
        Chunk(id=f"{page.doc_id}:p{page.page}:c{i}", doc_id=page.doc_id,
              source=page.source, page=page.page, text=body)
        for i, body in enumerate(bodies)
    ]


def chunk_pages(pages: list[Page], max_chars: int = 1200, overlap: int = 150) -> list[Chunk]:
    chunks: list[Chunk] = []
    for page in pages:
        chunks.extend(chunk_page(page, max_chars=max_chars, overlap=overlap))
    return chunks
