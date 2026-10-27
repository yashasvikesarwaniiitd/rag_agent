"""Streamlit chat.  uv run streamlit run app.py

The same file runs locally and on Streamlit Community Cloud. Settings come from .env locally
and from the app's Secrets on Cloud (root-level secrets are exposed as environment
variables). A public deployment should set RAG_ALLOW_UPLOADS=off and RAG_DAILY_BUDGET_USD.
"""
from __future__ import annotations

import os

import streamlit as st

from rag.adapters.loaders import SUPPORTED, FileLoader
from rag.application.ingest import find_files, ingest
from rag.config import (Settings, build_embedder, build_llm, build_reranker, build_rewriter,
                        build_service, build_store)
from rag.domain.budget import DailyBudget
from rag.domain.judging import cost_usd

st.set_page_config(page_title="Credentialing RAG", layout="wide")
settings = Settings.from_env()
HAS_KEY = bool(os.environ.get("ANTHROPIC_API_KEY"))


@st.cache_resource(show_spinner="Loading embedding model...")
def embedder():
    return build_embedder(settings)


@st.cache_resource(show_spinner="Loading re-ranker...")
def reranker():
    return build_reranker(settings)


@st.cache_resource
def store():
    return build_store(settings)


@st.cache_resource(show_spinner="First start: building the search index from the documents "
                                "(about a minute)...")
def ensure_index() -> int:
    """A fresh Cloud container starts with no index. Build it once from data/raw; the
    committed vector cache means no chunk has to be re-embedded."""
    if store().count() == 0:
        files = find_files(settings.data_dir)
        if files:
            ingest(files, loader=FileLoader(), embedder=embedder(), store=store())
    return store().count()


@st.cache_resource
def llm(choice: str):
    return build_llm(settings, choice)


@st.cache_resource
def rewriter(choice: str):
    # With the free model selected, only cached rewrites are used: picking "fake" means $0.
    return build_rewriter(settings, paid=choice == "anthropic") if settings.rewrite else None


@st.cache_resource
def budget() -> DailyBudget | None:
    # One instance shared by every visitor's session, so the cap is per app, not per tab.
    return DailyBudget(settings.daily_budget_usd) if settings.daily_budget_usd is not None else None


ensure_index()

with st.sidebar:
    st.header("Documents")
    st.caption(f"{store().count()} chunks indexed")
    for src in store().sources():
        st.write(f"- {src}")

    if settings.allow_uploads:
        uploads = st.file_uploader("Add documents", accept_multiple_files=True,
                                   type=[s.lstrip(".") for s in sorted(SUPPORTED)])
        if st.button("Index documents", type="primary"):
            settings.data_dir.mkdir(parents=True, exist_ok=True)
            for up in uploads or []:
                (settings.data_dir / up.name).write_bytes(up.getbuffer())
            files = find_files(settings.data_dir)
            if not files:
                st.warning(f"No documents in {settings.data_dir}")
            else:
                with st.spinner(f"Indexing {len(files)} files..."):
                    report = ingest(files, loader=FileLoader(), embedder=embedder(),
                                    store=store())
                st.success(f"{len(report.files)} files, {report.chunks} chunks")
                for skip in report.skipped:
                    st.warning(f"Skipped {skip}")

    st.header("Answer model")
    options = ["anthropic", "fake"] if HAS_KEY else ["fake"]
    default = "anthropic" if HAS_KEY and settings.llm == "anthropic" else "fake"
    captions = {"anthropic": f"{settings.model}, written answers",
                "fake": "Quotes the best passage, $0"}
    choice = st.radio("Model", options, index=options.index(default),
                      captions=[captions[o] for o in options])
    b = budget()
    if b is not None and choice == "anthropic":
        st.caption(f"Today's demo budget: ${b.spent:.3f} of ${b.limit:.2f} used")
    show_debug = st.toggle("Show retrieved chunks")

st.title("Credentialing RAG")
st.caption("Answers come only from the indexed public documents (CMS, eCFR, MACPAC), with "
           "citations. Education only, not legal or compliance advice.")

if "history" not in st.session_state:
    st.session_state.history = []


def render(answer, note: str = ""):
    if note:
        st.info(note)
    st.markdown(answer.text)
    if answer.refused:
        st.caption(f"Declined: {answer.reason}")
    if answer.rewrites:
        st.caption("Also searched as: " + " | ".join(answer.rewrites))
    for c in answer.citations:
        with st.expander(f"[{c.n}] {c.source}, page {c.page}"):
            if c.context:
                st.caption(c.context)
            st.text(c.snippet)
    if show_debug and answer.hits:
        with st.expander("Retrieved chunks"):
            for i, h in enumerate(answer.hits, start=1):
                st.markdown(f"**{i}.** `{h.score:.3f}` {h.chunk.source} p{h.chunk.page}")
                st.text(h.chunk.text[:500])
    st.caption(f"{answer.latency_ms} ms · {answer.input_tokens} in / {answer.output_tokens} "
               f"out tokens · {answer.model or 'no model call'}")


for question, answer, note in st.session_state.history:
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        render(answer, note)

if question := st.chat_input("Ask about credentialing requirements..."):
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        used, note = choice, ""
        b = budget()
        if used == "anthropic" and b is not None and not b.allows():
            used = "fake"
            note = ("Today's demo budget for written answers is used up, so this answer quotes "
                    "the best-matching passage instead. It resets at midnight UTC.")
        try:
            service = build_service(settings, embedder(), store(), llm(used),
                                    reranker=reranker(), rewriter=rewriter(used))
            with st.spinner("Searching documents..."):
                answer = service.ask(question)
        except Exception as exc:  # show the type, not internals, on a public page
            st.error(f"Something went wrong ({type(exc).__name__}). Please try again.")
        else:
            if used == "anthropic" and b is not None and answer.model:
                b.record(cost_usd(answer.model, answer.input_tokens, answer.output_tokens))
            render(answer, note)
            st.session_state.history.append((question, answer, note))
