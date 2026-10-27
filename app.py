"""Streamlit chat.  uv run streamlit run app.py"""
from __future__ import annotations

import streamlit as st

from rag.adapters.loaders import SUPPORTED, FileLoader
from rag.application.ingest import find_files, ingest
from rag.config import (Settings, build_embedder, build_llm, build_reranker, build_service,
                        build_store)

st.set_page_config(page_title="Credentialing RAG", layout="wide")
settings = Settings.from_env()


@st.cache_resource(show_spinner="Loading embedding model...")
def embedder():
    return build_embedder(settings)


@st.cache_resource(show_spinner="Loading re-ranker...")
def reranker():
    return build_reranker(settings)


@st.cache_resource
def store():
    return build_store(settings)


@st.cache_resource
def llm(choice: str):
    return build_llm(settings, choice)


with st.sidebar:
    st.header("Documents")
    st.caption(f"{store().count()} chunks indexed")
    for src in store().sources():
        st.write(f"- {src}")

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
                report = ingest(files, loader=FileLoader(), embedder=embedder(), store=store())
            st.success(f"{len(report.files)} files, {report.chunks} chunks")
            for skip in report.skipped:
                st.warning(f"Skipped {skip}")

    st.header("Answer model")
    choice = st.radio("Model", ["fake", "anthropic"],
                      index=0 if settings.llm == "fake" else 1,
                      captions=["Extractive, $0, no API key", f"{settings.model} (paid)"])
    show_debug = st.toggle("Show retrieved chunks")

st.title("Credentialing RAG")
st.caption("Answers come only from the indexed documents, with citations. "
           "Not legal or compliance advice.")

if "history" not in st.session_state:
    st.session_state.history = []


def render(answer):
    st.markdown(answer.text)
    if answer.refused:
        st.caption(f"Declined: {answer.reason}")
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


for question, answer in st.session_state.history:
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        render(answer)

if question := st.chat_input("Ask about credentialing requirements..."):
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        try:
            service = build_service(settings, embedder(), store(), llm(choice),
                                    reranker=reranker())
            with st.spinner("Searching documents..."):
                answer = service.ask(question)
        except Exception as exc:
            st.error(f"Something went wrong: {exc}")
        else:
            render(answer)
            st.session_state.history.append((question, answer))
