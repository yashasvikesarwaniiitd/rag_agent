"""File -> pages. PDFs via PyMuPDF; .txt/.md and .html (e.g. a saved eCFR section) as a
single page each, since they have no physical pages to cite.

PDF structure comes from the fonts: a line is a heading when it is bold or clearly larger than
the document's body text, and short enough to be a title. Lines repeated on many pages
(running headers/footers) and CMS revision stamps "(Rev. 24, 06-06-03)" are dropped.
"""
from __future__ import annotations

import re
from collections import Counter
from html.parser import HTMLParser
from pathlib import Path

from ..domain.chunking import NUMBERED_HEADING
from ..domain.models import Line, Page

SUPPORTED = {".pdf", ".txt", ".md", ".html", ".htm"}
_REV_STAMP = re.compile(r"^\(Rev\.[^)]*\)$")
_MAX_HEADING_CHARS = 120


def doc_id_for(path: Path) -> str:
    return path.stem.lower().replace(" ", "_")


# ---------------------------------------------------------------------------- HTML

class _HtmlExtractor(HTMLParser):
    _SKIP = {"script", "style", "nav", "header", "footer"}
    _BLOCK = {"p", "div", "li", "tr", "section", "br"}
    _HEADINGS = {"h1", "h2", "h3", "h4", "h5", "h6"}

    def __init__(self):
        super().__init__()
        self.lines: list[Line] = []
        self.title = ""
        self._buf: list[str] = []
        self._stack: list[tuple[str, bool]] = []  # (tag, skipping)
        self._in_heading = False
        self._pending_para: str | None = None

    def _skipping(self) -> bool:
        return any(skip for _, skip in self._stack)

    def _flush(self, heading: bool = False):
        text = re.sub(r"\s+", " ", "".join(self._buf)).strip()
        self._buf = []
        if text:
            self.lines.append(Line(text, heading=heading))
            if heading and not self.title:
                self.title = text.rstrip(".")
        self.lines.append(Line(""))

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        # eCFR wraps clutter like the part-level "Editorial Note" in a classed div.
        skip = tag in self._SKIP or "editorial-note" in (attrs.get("class") or "")
        if tag not in {"br", "img", "hr", "input", "meta", "link"}:
            self._stack.append((tag, skip))
        if self._skipping():
            return
        if tag in self._HEADINGS:
            self._flush()
            self._in_heading = True
        elif tag in self._BLOCK:
            self._flush()
            # eCFR top-level paragraphs: <p class="indent-1" data-title="422.204(a)">.
            if tag == "p" and "indent-1" in (attrs.get("class") or "") and attrs.get("data-title"):
                self._pending_para = attrs["data-title"]

    def handle_endtag(self, tag):
        if not self._skipping():
            if tag in self._HEADINGS:
                self._flush(heading=True)
                self._in_heading = False
            elif tag == "em" and self._pending_para:
                # "(a) General rule." becomes a heading line; the rest of the paragraph is body.
                label = re.sub(r"\s+", " ", "".join(self._buf)).strip().rstrip(".")
                self._buf = []
                if label:
                    self.lines.append(Line(label, heading=True))
                self._pending_para = None
            elif tag in self._BLOCK:
                self._flush()
        while self._stack:
            open_tag, _ = self._stack.pop()
            if open_tag == tag:
                break

    def handle_data(self, data):
        if not self._skipping():
            self._buf.append(data)

    def close(self):
        super().close()
        self._flush()


def html_to_text(html: str) -> str:
    parser = _HtmlExtractor()
    parser.feed(html)
    parser.close()
    return "\n".join(line.text for line in parser.lines)


def _load_html(path: Path, doc_id: str) -> list[Page]:
    parser = _HtmlExtractor()
    parser.feed(path.read_text(encoding="utf-8", errors="replace"))
    parser.close()
    lines = tuple(parser.lines)
    return [Page(doc_id=doc_id, source=path.name, page=1,
                 text="\n".join(line.text for line in lines), lines=lines,
                 doc_title=parser.title)]


# ----------------------------------------------------------------------------- PDF

def _norm_repeat(text: str) -> str:
    return re.sub(r"\d+", "#", text.strip().lower())


def _pdf_lines(pdf) -> tuple[list[list[tuple[str, float, bool]]], float]:
    """Per page: (text, max font size, all-bold) per line, with blank entries between
    blocks; plus the document's body font size (the most common size, weighted by chars)."""
    sizes: Counter = Counter()
    pages = []
    for page in pdf:
        out: list[tuple[str, float, bool]] = []
        for block in page.get_text("dict")["blocks"]:
            for line in block.get("lines", []):
                spans = [s for s in line["spans"] if s["text"].strip()]
                if not spans:
                    continue
                text = "".join(s["text"] for s in line["spans"]).strip()
                for s in spans:
                    sizes[round(s["size"], 1)] += len(s["text"])
                out.append((text, max(s["size"] for s in spans),
                            all(s["flags"] & 16 or "bold" in s["font"].lower() for s in spans)))
            out.append(("", 0.0, False))
        pages.append(out)
    body = sizes.most_common(1)[0][0] if sizes else 10.0
    return pages, body


def _title_from(pdf, first_page: list[tuple[str, float, bool]]) -> str:
    meta = (pdf.metadata or {}).get("title", "").strip()
    if meta and not meta.lower().startswith(("microsoft", "untitled")) and len(meta) > 8:
        return meta
    if not first_page:
        return ""
    top = max(size for _, size, _ in first_page)
    return " ".join(t for t, size, _ in first_page if t and size == top)[:150]


def _load_pdf(path: Path, doc_id: str) -> list[Page]:
    import pymupdf  # imported here so non-PDF users and tests don't pay for it

    with pymupdf.open(path) as pdf:
        raw, body = _pdf_lines(pdf)
        title = _title_from(pdf, raw[0] if raw else [])

    # Running headers/footers: the same line (digits ignored) on many pages.
    repeated: set[str] = set()
    if len(raw) >= 5:
        seen = Counter(k for page in raw for k in {_norm_repeat(t) for t, _, _ in page if t})
        repeated = {k for k, n in seen.items() if n >= max(3, 0.3 * len(raw))}

    pages = []
    for number, page_lines in enumerate(raw, start=1):
        lines: list[Line] = []
        for text, size, bold in page_lines:
            if text and (_norm_repeat(text) in repeated or _REV_STAMP.match(text)):
                continue
            is_heading = bool(text) and len(text) <= _MAX_HEADING_CHARS and (
                NUMBERED_HEADING.match(text) is not None
                or ((bold or size >= body + 1.5) and not text.endswith((".", ",", ";", ":"))
                    and any(c.isalpha() for c in text)))
            lines.append(Line(text, heading=is_heading))
        pages.append(Page(doc_id=doc_id, source=path.name, page=number,
                          text="\n".join(line.text for line in lines), lines=tuple(lines),
                          doc_title=title))
    return pages


class FileLoader:
    def load(self, path: Path) -> list[Page]:
        path = Path(path)
        suffix = path.suffix.lower()
        doc_id = doc_id_for(path)
        if suffix == ".pdf":
            return _load_pdf(path, doc_id)
        if suffix in {".txt", ".md"}:
            return [Page(doc_id=doc_id, source=path.name, page=1,
                         text=path.read_text(encoding="utf-8", errors="replace"))]
        if suffix in {".html", ".htm"}:
            return _load_html(path, doc_id)
        raise ValueError(f"unsupported file type: {path.name}")
