"""CLI.

    uv run python -m rag ingest            # index everything in data/raw
    uv run python -m rag ask "question"    # answer with citations
    uv run python -m rag stats             # what is indexed
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .application.answer import RagService
from .application.ingest import find_files, ingest
from .adapters.loaders import FileLoader
from .config import Settings, build_embedder, build_llm, build_store


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="rag")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_ingest = sub.add_parser("ingest", help="index documents")
    p_ingest.add_argument("paths", nargs="*", type=Path, help="files (default: data/raw)")
    p_ask = sub.add_parser("ask", help="ask a question")
    p_ask.add_argument("question")
    p_ask.add_argument("--llm", choices=["anthropic", "fake"], help="override RAG_LLM")
    p_ask.add_argument("--show-sources", action="store_true")
    sub.add_parser("stats", help="show the index")
    args = parser.parse_args(argv)

    s = Settings.from_env()
    store = build_store(s)

    if args.cmd == "stats":
        print(f"{store.count()} chunks from {len(store.sources())} documents")
        for src in store.sources():
            print(f"  - {src}")
        return 0

    if args.cmd == "ingest":
        paths = args.paths or find_files(s.data_dir)
        if not paths:
            print(f"No documents found in {s.data_dir}", file=sys.stderr)
            return 1
        report = ingest(paths, loader=FileLoader(), embedder=build_embedder(s), store=store)
        print(f"Indexed {len(report.files)} files, {report.pages} pages, {report.chunks} chunks")
        if report.empty_pages:
            print(f"  {report.empty_pages} pages had no text")
        for skip in report.skipped:
            print(f"  skipped {skip}")
        return 0

    if store.count() == 0:
        # Checked before building the embedder, which loads (and on first use downloads) a model.
        print("No documents are indexed yet. Run: uv run python -m rag ingest", file=sys.stderr)
        return 1
    service = RagService(embedder=build_embedder(s), store=store,
                         llm=build_llm(s, args.llm), top_k=s.top_k)
    answer = service.ask(args.question)
    print(answer.text)
    if answer.refused:
        print(f"\n(declined: {answer.reason})")
    for c in answer.citations:
        print(f"  [{c.n}] {c.source}, page {c.page}")
    if args.show_sources:
        print("\nRetrieved:")
        for i, h in enumerate(answer.hits, start=1):
            print(f"  {i}. {h.score:.3f}  {h.chunk.source} p{h.chunk.page}")
    print(f"\n{answer.latency_ms} ms | {answer.input_tokens} in / {answer.output_tokens} out "
          f"tokens | {answer.model or 'no model call'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
