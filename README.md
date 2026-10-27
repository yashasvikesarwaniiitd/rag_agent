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

Indexing the 5 sample documents (3,823 chunks) takes about 7 minutes the first time on a
4-core CPU. Vectors are cached in `data/cache/` by model + exact chunk text, so later
re-indexes only embed new or edited chunks. A full rebuild with nothing changed takes about
30 seconds, most of it loading PyTorch.

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

Since v0.1, retrieval is three stages:

1. **Vector search** (local bge-small) and **BM25 keyword search**, 30 candidates each.
2. Merged by reciprocal rank fusion.
3. The top 25 are **re-ranked** by a local cross-encoder (`ms-marco-MiniLM-L6-v2`, ~90MB).
   It reads each chunk's section label, e.g. "§ 422.204 … > (b) Basic requirements",
   alongside the text.

Chunks follow the document's headings. Running headers, revision stamps and the eCFR
editorial note are dropped before chunking.

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

## Evaluation

`evals/golden.jsonl` holds 20 answerable questions (11 in policy wording, 9 in plain English),
each with the page(s) that answer it, plus 2 that should be declined. Scoring is free, with
no model calls:

```bash
uv run python evals/run_retrieval.py --modes vector hybrid hybrid+rerank
```

| Step | Found in top 5 | MRR | Plain English |
|---|---|---|---|
| v0 chunks, vector only | 16/20 | 0.64 | 5/9 |
| v0 chunks, hybrid | 16/20 | 0.68 | 5/9 |
| v0.1 chunks, section labels **in** the index | 14/20 | 0.60 | 4/9 |
| v0.1 chunks, labels kept out of the index | 16/20 | 0.64 | 5/9 |
| **+ re-ranker reading the labels (current)** | **18/20** | **0.82** | **7/9** |

MRR is 1.0 when the right page always comes first, and 0.2 when it comes fifth. These
numbers come from 20 questions that were also used to pick settings, so differences of
one question are noise. Full per-question tables are in `evals/results/`.

Still failing: "Should our plan approve Dr. Patel?" is answered instead of declined, which
needs the router (v0.3).

### Answer quality (Haiku 5.5)

`evals/run_answers.py` asks every golden question, then grades each answer three ways:

- **Correct:** the expected fact (e.g. "3 years") appears in the answer. This is a free string check.
- **Faithful:** every claim is supported by the sources, judged by Haiku.
- **On topic:** the answer addresses this question, not a neighbouring rule. Also judged by Haiku.

```bash
uv run python evals/run_answers.py --dry-run        # $0, cost estimate from the real prompts
uv run python evals/run_answers.py --fake           # $0, whole loop with fake models
uv run python evals/run_answers.py --limit 3        # paid, ~$0.002-0.005
uv run python evals/run_answers.py --save v0.2      # paid, all 22: estimated $0.012-$0.038
```

Paid runs default to effort `low` and stop once `--max-cost` (default $0.05) has been spent,
using each response's real token counts. The judge is the same model that wrote the answers,
so read its "faithful" score as an upper bound.

**Windows note:** stop the Streamlit app before re-indexing. It holds the index files open,
and the delete fails with "Device or resource busy".

## Roadmap

| Version | Adds | Diagram box |
|---|---|---|
| **v0.0** | basic RAG, citation gate, CLI + Streamlit | Data Sources -> Vector Store -> answer |
| **v0.1** | hybrid keyword search, golden set + retrieval scorer, heading-aware chunks, re-ranker | Structure-Aware Chunking, Evaluation |
| v0.2 | faithfulness judge on Haiku answers, latency & cost tracking, table preservation | Evaluation |
| v0.3 | router (in/out of scope), prompt-injection tests | Conditional Router, Stress Testing |
| v0.4 | per-chunk summary, keywords, synthetic questions | Metadata Creation |
| later | planner, multi-agent, human review queue (only if measured need) | Reasoning Engine, Human Validation |
