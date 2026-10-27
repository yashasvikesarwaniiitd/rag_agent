"""Ingest: files -> pages -> chunks -> vectors -> store."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from ..adapters.loaders import SUPPORTED
from ..domain.chunking import chunk_pages
from ..ports import Embedder, Loader, VectorStore


@dataclass
class IngestReport:
    files: list[str] = field(default_factory=list)
    pages: int = 0
    chunks: int = 0
    empty_pages: int = 0
    skipped: list[str] = field(default_factory=list)  # "name: reason"


def find_files(folder: Path) -> list[Path]:
    return sorted(p for p in Path(folder).rglob("*")
                  if p.is_file() and p.suffix.lower() in SUPPORTED)


def ingest(paths: list[Path], *, loader: Loader, embedder: Embedder, store: VectorStore,
           max_chars: int = 1200, overlap: int = 150, batch_size: int = 64) -> IngestReport:
    report = IngestReport()
    for path in paths:
        try:
            pages = loader.load(path)
        except Exception as exc:  # one bad file should not stop the rest
            report.skipped.append(f"{Path(path).name}: {exc}")
            continue
        chunks = chunk_pages(pages, max_chars=max_chars, overlap=overlap)
        report.pages += len(pages)
        report.empty_pages += sum(1 for p in pages if not p.text.strip())
        if not chunks:
            # Usually a scanned PDF with no text layer - needs OCR, which v0 doesn't do.
            report.skipped.append(f"{Path(path).name}: no extractable text (scanned PDF?)")
            continue
        # Re-ingesting a file replaces it, so an edited document leaves no stale chunks.
        store.delete_doc(chunks[0].doc_id)
        for start in range(0, len(chunks), batch_size):
            batch = chunks[start:start + batch_size]
            store.upsert(batch, embedder.embed_documents([c.text for c in batch]))
        report.files.append(Path(path).name)
        report.chunks += len(chunks)
    return report
