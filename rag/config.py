"""Settings from the environment (.env), and the one place real adapters are wired up."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

from .adapters.embedders import DEFAULT_EMBED_MODEL
from .adapters.llms import DEFAULT_MODEL
from .adapters.rerankers import DEFAULT_RERANK_MODEL

ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Settings:
    data_dir: Path
    index_dir: Path
    embed_model: str
    llm: str  # "anthropic" | "fake"
    model: str
    top_k: int
    retrieval: str  # "hybrid" | "vector"
    candidates: int
    min_score: float
    rerank_model: str  # "" = no re-ranker
    rerank_pool: int
    rewrite: bool

    @classmethod
    def from_env(cls) -> "Settings":
        load_dotenv(ROOT / ".env")
        return cls(
            data_dir=ROOT / os.environ.get("RAG_DATA_DIR", "data/raw"),
            index_dir=ROOT / os.environ.get("RAG_INDEX_DIR", "data/index"),
            embed_model=os.environ.get("RAG_EMBED_MODEL", DEFAULT_EMBED_MODEL),
            llm=os.environ.get("RAG_LLM", "fake").lower(),
            model=os.environ.get("RAG_MODEL", DEFAULT_MODEL),
            top_k=int(os.environ.get("RAG_TOP_K", "5")),
            retrieval=os.environ.get("RAG_RETRIEVAL", "hybrid").lower(),
            candidates=int(os.environ.get("RAG_CANDIDATES", "30")),
            # From 7 probe questions: real ones scored 0.72-0.88, "capital of France" 0.43.
            # Provisional until the golden set gives enough questions to tune it properly.
            min_score=float(os.environ.get("RAG_MIN_SCORE", "0.6")),
            # On by default: golden set MRR 0.64 -> ~0.8 for ~2s per question on CPU.
            # Set RAG_RERANK_MODEL= (empty) to turn it off.
            rerank_model=os.environ.get("RAG_RERANK_MODEL", DEFAULT_RERANK_MODEL),
            # 25 is chosen for speed; 25/40/60 scored within one question of each other.
            rerank_pool=int(os.environ.get("RAG_RERANK_POOL", "25")),
            # Off until measured on the golden set. Costs ~$0.0001-0.0002 per question.
            rewrite=os.environ.get("RAG_REWRITE", "off").lower() in {"on", "1", "true"},
        )


def build_rewriter(s: Settings, paid: bool):
    """Cached query rewriter. paid=False never calls the API: only rewrites already in the
    cache are used, so the $0 paths (fake model, free evals) stay $0."""
    from .adapters.rewriters import CachingRewriter, LLMQueryRewriter
    inner = None
    if paid:
        from .adapters.llms import AnthropicLLM
        inner = LLMQueryRewriter(AnthropicLLM(model=s.model, effort="low"))
    return CachingRewriter(inner, ROOT / "data" / "cache" / "rewrites.sqlite", s.model,
                           paid=paid)


def build_reranker(s: Settings, model: str | None = None):
    name = s.rerank_model if model is None else model
    if not name:
        return None
    from .adapters.rerankers import CrossEncoderReranker
    return CrossEncoderReranker(name)


def build_service(s: Settings, embedder, store, llm, reranker=None, rewriter=None):
    from .application.answer import RagService
    return RagService(embedder=embedder, store=store, llm=llm, top_k=s.top_k,
                      retrieval=s.retrieval, candidates=s.candidates, min_score=s.min_score,
                      reranker=reranker, rerank_pool=s.rerank_pool, rewriter=rewriter)


def build_embedder(s: Settings):
    from .adapters.embedders import SentenceTransformerEmbedder
    from .adapters.embedding_cache import CachingEmbedder
    return CachingEmbedder(SentenceTransformerEmbedder(s.embed_model),
                           ROOT / "data" / "cache" / "embeddings.sqlite", s.embed_model)


def build_store(s: Settings):
    from .adapters.stores import ChromaStore
    return ChromaStore(s.index_dir)


def build_llm(s: Settings, llm: str | None = None):
    from .adapters.llms import AnthropicLLM, ExtractiveLLM
    choice = (llm or s.llm).lower()
    if choice == "anthropic":
        return AnthropicLLM(model=s.model)
    if choice == "fake":
        return ExtractiveLLM()
    raise ValueError(f"RAG_LLM must be 'anthropic' or 'fake', got {choice!r}")
