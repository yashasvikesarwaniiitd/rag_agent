"""Splitting documents into chunks. Pure functions.

Structure-aware since v0.1:
- a heading always starts a new chunk, so one chunk never straddles two sections;
- the current section carries across page breaks, and every chunk is labelled
  "document title > numbered section > sub-heading" (Chunk.context);
- within a section, paragraphs are packed up to `max_chars` with a character overlap.
Chunks still never cross a page boundary, which keeps every citation to ONE page.
"""
from __future__ import annotations

import re

from .models import Chunk, Line, Page

_PARA_SPLIT = re.compile(r"\n\s*\n")
_WS = re.compile(r"[ \t]+")
# Manual section numbers: "60.3 - Credentialing...", "10.4.5 – Revalidations".
NUMBERED_HEADING = re.compile(r"^\d{1,3}(?:\.\d{1,3})*\s+[-–—]\s+\S.{0,150}$")


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


def pack(text: str, max_chars: int, overlap: int) -> list[str]:
    """Pack the paragraphs of one block of text into bodies of at most max_chars."""
    if overlap >= max_chars:
        raise ValueError("overlap must be smaller than max_chars")
    text = normalise(text)
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
    return bodies


def _lines_of(page: Page) -> tuple[Line, ...]:
    """Structured lines if the loader gave them, else plain lines with numbered headings."""
    if page.lines:
        return page.lines
    return tuple(Line(ln, heading=bool(NUMBERED_HEADING.match(ln.strip())))
                 for ln in page.text.replace("\r", "").split("\n"))


class _Breadcrumb:
    """Tracks "numbered section > latest sub-heading" as the document is read in order."""

    def __init__(self, title: str):
        self.title, self.section, self.sub = title, "", ""

    def see(self, heading: str) -> None:
        heading = _WS.sub(" ", heading).strip()
        in_numbered_section = bool(NUMBERED_HEADING.match(self.section))
        # "(a) General rule" is a top-level section in a regulation, but inside a manual's
        # "10.4.5 - Revalidations" a bold "(iii)" is only a sub-heading.
        paragraph_marker = re.match(r"^\(\w+\)", heading) and not in_numbered_section
        if NUMBERED_HEADING.match(heading) or heading.startswith("§") or paragraph_marker:
            self.section, self.sub = heading, ""
        else:
            self.sub = heading

    def label(self) -> str:
        return " > ".join(x for x in (self.title, self.section, self.sub) if x)


def chunk_document(pages: list[Page], max_chars: int = 1200, overlap: int = 150) -> list[Chunk]:
    """Pages of ONE document, in page order."""
    if not pages:
        return []
    crumb = _Breadcrumb(pages[0].doc_title)
    chunks: list[Chunk] = []
    for page in pages:
        segments: list[tuple[str, list[str]]] = [(crumb.label(), [])]
        for line in _lines_of(page):
            if line.heading and line.text.strip():
                crumb.see(line.text)
                segments.append((crumb.label(), []))
            else:
                segments[-1][1].append(line.text)
        i = 0
        for label, body_lines in segments:
            for body in pack("\n".join(body_lines), max_chars, overlap):
                chunks.append(Chunk(id=f"{page.doc_id}:p{page.page}:c{i}", doc_id=page.doc_id,
                                    source=page.source, page=page.page, text=body,
                                    context=label))
                i += 1
    return chunks


def chunk_page(page: Page, max_chars: int = 1200, overlap: int = 150) -> list[Chunk]:
    return chunk_document([page], max_chars=max_chars, overlap=overlap)


def chunk_pages(pages: list[Page], max_chars: int = 1200, overlap: int = 150) -> list[Chunk]:
    """Pages possibly from several documents; each document is chunked in page order."""
    by_doc: dict[str, list[Page]] = {}
    for page in pages:
        by_doc.setdefault(page.doc_id, []).append(page)
    chunks: list[Chunk] = []
    for doc_pages in by_doc.values():
        chunks.extend(chunk_document(sorted(doc_pages, key=lambda p: p.page), max_chars, overlap))
    return chunks
