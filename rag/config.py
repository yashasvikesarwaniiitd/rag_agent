"""Settings from the environment (.env), and the one place real adapters are wired up."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

from .adapters.embedders import DEFAULT_EMBED_MODEL
from .adapters.llms import DEFAULT_MODEL

ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Settings:
    data_dir: Path
    index_dir: Path
    embed_model: str
    llm: str  # "anthropic" | "fake"
    model: str
    top_k: int

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
        )


def build_embedder(s: Settings):
    from .adapters.embedders import SentenceTransformerEmbedder
    return SentenceTransformerEmbedder(s.embed_model)


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
