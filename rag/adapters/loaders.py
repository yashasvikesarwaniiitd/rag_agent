"""File -> pages. PDFs via PyMuPDF; .txt/.md and .html (e.g. a saved eCFR section) as a
single page each, since they have no physical pages to cite."""
from __future__ import annotations

from html.parser import HTMLParser
from pathlib import Path

from ..domain.models import Page

SUPPORTED = {".pdf", ".txt", ".md", ".html", ".htm"}


def doc_id_for(path: Path) -> str:
    return path.stem.lower().replace(" ", "_")


class _TextExtractor(HTMLParser):
    _SKIP = {"script", "style", "nav", "header", "footer"}
    _BLOCK = {"p", "div", "li", "tr", "h1", "h2", "h3", "h4", "h5", "h6", "br", "section"}

    def __init__(self):
        super().__init__()
        self.parts: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag, attrs):
        if tag in self._SKIP:
            self._skip_depth += 1
        elif tag in self._BLOCK:
            self.parts.append("\n\n")

    def handle_endtag(self, tag):
        if tag in self._SKIP and self._skip_depth:
            self._skip_depth -= 1

    def handle_data(self, data):
        if not self._skip_depth:
            self.parts.append(data)


def html_to_text(html: str) -> str:
    parser = _TextExtractor()
    parser.feed(html)
    return "".join(parser.parts)


class FileLoader:
    def load(self, path: Path) -> list[Page]:
        path = Path(path)
        suffix = path.suffix.lower()
        doc_id = doc_id_for(path)
        if suffix == ".pdf":
            import pymupdf  # imported here so non-PDF users and tests don't pay for it

            with pymupdf.open(path) as pdf:
                return [Page(doc_id=doc_id, source=path.name, page=i + 1,
                             text=page.get_text("text"))
                        for i, page in enumerate(pdf)]
        if suffix in {".txt", ".md"}:
            return [Page(doc_id=doc_id, source=path.name, page=1,
                         text=path.read_text(encoding="utf-8", errors="replace"))]
        if suffix in {".html", ".htm"}:
            html = path.read_text(encoding="utf-8", errors="replace")
            return [Page(doc_id=doc_id, source=path.name, page=1, text=html_to_text(html))]
        raise ValueError(f"unsupported file type: {path.name}")
