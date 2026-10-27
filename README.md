# rag_agent: Credentialing RAG

Ask questions about healthcare provider credentialing and get answers drawn **only** from
public policy documents (CMS manuals, federal regulations, state Medicaid policies), each
claim cited to a document and page.

Analytics and education only, not legal or compliance advice.

## Quickstart

```bash
uv sync
cp .env.example .env          # defaults to the $0 extractive model
# put PDFs / .html / .txt files into data/raw/
uv run python -m rag ingest
uv run python -m rag ask "How often must network providers be recredentialed?" --show-sources
uv run streamlit run app.py
```

The first ingest downloads the local embedding model (`BAAI/bge-small-en-v1.5`, ~130MB)
once. To use Claude Haiku 5.5 for answers, set `ANTHROPIC_API_KEY` and `RAG_LLM=anthropic`
in `.env`, or pick it in the Streamlit sidebar.

## How v0 works

```
data/raw/*.pdf|html|txt
   │  FileLoader (PyMuPDF)          -> pages, 1-based page numbers
   │  chunk_pages                   -> ~1,200-char chunks, 150 overlap, never crossing a page
   │  SentenceTransformerEmbedder   -> local vectors, nothing leaves the machine
   ▼
ChromaStore (data/index/, cosine)
   ▲
question -> embed -> top-k chunks -> numbered <source> prompt -> LLM
                                                                │
                                         citation gate ◄────────┘
                       (every [n] must be a retrieved source, or answer NOT_FOUND)
                                         │
                              Answer + citations (doc, page)
```

An answer that cites nothing, or cites a source that wasn't retrieved, is **never shown**.
The user gets a decline with the reason instead. The prompt asks the model to cite, and the
gate makes sure it did.

## Layout

| Path | What | Imports SDKs? |
|---|---|---|
| `rag/domain/` | data models, chunker, citation gate: pure functions | no |
| `rag/ports.py` | interfaces: Loader, Embedder, VectorStore, LLM | no |
| `rag/adapters/` | PyMuPDF, sentence-transformers, Chroma, Anthropic, plus fakes | yes |
| `rag/application/` | ingest and answer services, prompts | no |
| `rag/config.py` | `.env` settings and the wiring of real adapters | - |
| `app.py`, `rag/__main__.py` | Streamlit chat, CLI | - |

## Tests

```bash
uv run pytest
```

All tests run offline at $0: a hashing embedder, an in-memory store and a scripted model
stand in for the real ones. PyMuPDF and Chroma are exercised for real because they are local.

## Roadmap

| Version | Adds | Diagram box |
|---|---|---|
| **v0.0** | basic RAG, citation gate, CLI + Streamlit | Data Sources -> Vector Store -> answer |
| v0.1 | heading detection, table preservation, hybrid keyword search | Structure-Aware Chunking |
| v0.2 | golden set, retrieval precision/recall, faithfulness judge, latency & cost | Evaluation |
| v0.3 | router (in/out of scope), prompt-injection tests | Conditional Router, Stress Testing |
| v0.4 | per-chunk summary, keywords, synthetic questions | Metadata Creation |
| later | planner, multi-agent, human review queue (only if measured need) | Reasoning Engine, Human Validation |
