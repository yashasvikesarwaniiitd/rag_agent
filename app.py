"""Streamlit chat.  uv run streamlit run app.py

The same file runs locally and on Streamlit Community Cloud. Settings come from .env locally
and from the app's Secrets on Cloud (root-level secrets are exposed as environment
variables). A public deployment should set RAG_ALLOW_UPLOADS=off and RAG_DAILY_BUDGET_USD.
"""
from __future__ import annotations

import html
import os
import re

import streamlit as st

from rag.adapters.loaders import SUPPORTED, FileLoader
from rag.application.ingest import find_files, ingest
from rag.config import (Settings, build_embedder, build_llm, build_reranker, build_rewriter,
                        build_service, build_store)
from rag.domain.budget import DailyBudget
from rag.domain.judging import cost_usd

st.set_page_config(page_title="Credentialing RAG Agent", page_icon="🩺", layout="wide")
settings = Settings.from_env()
HAS_KEY = bool(os.environ.get("ANTHROPIC_API_KEY"))

# Friendly names for the corpus; unknown files fall back to their file name.
DOC_NAMES = {
    "cms_medicare_managed_care_ch6": ("Medicare Managed Care Manual, Ch. 6", "CMS manual"),
    "cms_pim_ch10": ("Medicare Program Integrity Manual, Ch. 10", "CMS manual"),
    "ecfr_42cfr_422.204": ("42 CFR § 422.204: MA provider selection", "Regulation"),
    "ecfr_42cfr_438.214": ("42 CFR § 438.214: Medicaid provider selection", "Regulation"),
    "macpac_ch7": ("MACPAC Report, Ch. 7 (June 2026)", "Analysis"),
}
EXAMPLES = [
    "How often must an MA organization recredential physicians?",
    "What must a state's uniform credentialing policy cover under 42 CFR 438.214?",
    "How long does a Medicaid provider have to submit fingerprints?",
    "Which health care professionals don't need to be credentialed by an MA plan?",
]

st.markdown("""
<style>
.block-container {padding-top: 2rem; max-width: 1100px;}
.hero h1 {font-size: 2.1rem; margin-bottom: .2rem;}
.hero p {color: #5b6473; margin-top: 0;}
.pill {display:inline-block; padding:2px 10px; margin:0 6px 6px 0; border-radius:999px;
       font-size:.78rem; background:#eef2ff; color:#3730a3; border:1px solid #e0e7ff;}
.pill.ok {background:#ecfdf5; color:#065f46; border-color:#d1fae5;}
.pill.warn {background:#fff7ed; color:#9a3412; border-color:#ffedd5;}
.cite {display:inline-block; min-width:1.35em; padding:0 .35em; margin:0 1px;
       border-radius:6px; font-size:.75em; font-weight:600; text-align:center;
       background:#e0e7ff; color:#3730a3; vertical-align:super;}
.src {border:1px solid #e5e7eb; border-radius:10px; padding:.6rem .8rem; margin:.4rem 0;
      background:#fafafa;}
.src .meta {font-size:.8rem; color:#6b7280;}
.src .quote {font-size:.85rem; color:#374151; margin-top:.3rem;}
.muted {color:#6b7280; font-size:.8rem;}
</style>
""", unsafe_allow_html=True)


def doc_label(doc_id: str, source: str) -> tuple[str, str]:
    for prefix, names in DOC_NAMES.items():
        if doc_id.startswith(prefix) or source.lower().startswith(prefix):
            return names
    return source, "Document"


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

# ------------------------------------------------------------------------------- sidebar
with st.sidebar:
    st.markdown("### 🩺 Credentialing RAG Agent")
    st.caption("Cited answers from federal credentialing rules")

    st.markdown("**Answer engine**")
    options = ["anthropic", "fake"] if HAS_KEY else ["fake"]
    default = "anthropic" if HAS_KEY and settings.llm == "anthropic" else "fake"
    labels = {"anthropic": f"Claude ({settings.model})", "fake": "Quote only (free)"}
    choice = st.radio("Answer engine", options, index=options.index(default),
                      format_func=labels.get, label_visibility="collapsed")
    b = budget()
    if b is not None and choice == "anthropic":
        st.progress(min(b.spent / b.limit, 1.0) if b.limit else 1.0,
                    text=f"Today's budget: ${b.spent:.3f} of ${b.limit:.2f}")

    st.markdown("**Knowledge base**")
    by_doc: dict[str, int] = {}
    for c in store().all_chunks():
        by_doc[c.source] = by_doc.get(c.source, 0) + 1
    for source, n in sorted(by_doc.items()):
        name, kind = doc_label(source.rsplit(".", 1)[0].lower(), source)
        st.markdown(f"<div class='src'><b>{html.escape(name)}</b><div class='meta'>"
                    f"{kind} · {n:,} passages</div></div>", unsafe_allow_html=True)

    if settings.allow_uploads:
        with st.expander("Add documents"):
            uploads = st.file_uploader("Files", accept_multiple_files=True,
                                       type=[s.lstrip(".") for s in sorted(SUPPORTED)],
                                       label_visibility="collapsed")
            if st.button("Index documents", type="primary", use_container_width=True):
                settings.data_dir.mkdir(parents=True, exist_ok=True)
                for up in uploads or []:
                    (settings.data_dir / up.name).write_bytes(up.getbuffer())
                files = find_files(settings.data_dir)
                with st.spinner(f"Indexing {len(files)} files..."):
                    report = ingest(files, loader=FileLoader(), embedder=embedder(),
                                    store=store())
                st.success(f"{len(report.files)} files, {report.chunks} chunks")
                for skip in report.skipped:
                    st.warning(f"Skipped {skip}")

    with st.expander("How it works"):
        st.markdown(
            "1. **Search**: meaning-based and keyword search over every passage\n"
            "2. **Re-rank**: a second model reads the question with the top 25 passages\n"
            "3. **Answer**: Claude writes only from the 5 best passages\n"
            "4. **Verify**: every `[n]` must point at a retrieved passage, or the answer "
            "is withheld\n\n"
            "It says *not found* rather than guess, and won't make decisions about "
            "specific providers.")
    show_debug = st.toggle("Show retrieval details")
    if st.button("Clear conversation", use_container_width=True):
        st.session_state.history = []
        st.rerun()

# ------------------------------------------------------------------------------- main
st.markdown(
    "<div class='hero'><h1>Ask about provider credentialing</h1>"
    "<p>Answers come only from the indexed public documents, with a citation on every "
    "claim. Education only, not legal or compliance advice.</p></div>",
    unsafe_allow_html=True)
pills = [f"{len(by_doc)} documents", f"{sum(by_doc.values()):,} passages",
         "Hybrid search + re-ranker", "Citation-checked"]
st.markdown("".join(f"<span class='pill'>{p}</span>" for p in pills), unsafe_allow_html=True)

if "history" not in st.session_state:
    st.session_state.history = []


def with_cite_chips(text: str) -> str:
    """Escape the model's text first (it is untrusted), then style [1] / [2, 3] markers."""
    safe = html.escape(text)
    return re.sub(r"\[(\d+(?:\s*,\s*\d+)*)\]",
                  lambda m: "".join(f"<span class='cite'>{n.strip()}</span>"
                                    for n in m.group(1).split(",")), safe).replace("\n", "<br>")


def render(answer, note: str = ""):
    if note:
        st.info(note, icon="💡")
    if answer.refused:
        icon = "🔎" if "cover" in answer.reason or "off-topic" in answer.reason else "🛡️"
        st.warning(answer.text, icon=icon)
        st.caption(f"Why: {answer.reason}")
    else:
        st.markdown(f"<div style='font-size:1.02rem; line-height:1.6'>"
                    f"{with_cite_chips(answer.text)}</div>", unsafe_allow_html=True)

    if answer.citations:
        st.markdown("<div class='muted' style='margin-top:.8rem'>SOURCES</div>",
                    unsafe_allow_html=True)
        for c in answer.citations:
            name, kind = doc_label(c.chunk_id.split(":")[0], c.source)
            section = c.context.split(" > ")[-1] if c.context else ""
            st.markdown(
                f"<div class='src'><span class='cite'>{c.n}</span> <b>{html.escape(name)}</b>"
                f"<div class='meta'>{kind} · page {c.page}"
                f"{' · ' + html.escape(section) if section else ''}</div>"
                f"<div class='quote'>“{html.escape(' '.join(c.snippet.split())[:260])}…”</div>"
                f"</div>", unsafe_allow_html=True)

    steps = []
    if answer.hits:
        steps.append(f"🔍 Retrieved {len(answer.hits)} passages "
                     f"from {len({h.chunk.source for h in answer.hits})} documents")
    if answer.rewrites:
        steps.append("✏️ Also searched as: " + " | ".join(answer.rewrites))
    if answer.model:
        steps.append(f"🧠 Answered by {answer.model}")
    if answer.citations:
        steps.append(f"✅ {len(answer.citations)} citation(s) verified against the retrieved "
                     "passages")
    cost = cost_usd(answer.model, answer.input_tokens, answer.output_tokens) if answer.model else 0
    steps.append(f"⏱️ {answer.latency_ms / 1000:.1f}s"
                 + (f" · ${cost:.4f}" if cost else " · $0"))
    with st.expander("Agent trace", expanded=False):
        for s in steps:
            st.markdown(f"- {s}")
        if show_debug and answer.hits:
            st.markdown("**Retrieved passages (re-ranked)**")
            for i, h in enumerate(answer.hits, start=1):
                st.markdown(f"`{i}` · score `{h.score:.2f}` · {html.escape(h.chunk.source)} "
                            f"p{h.chunk.page}")
                st.caption(" ".join(h.chunk.text.split())[:400])


def ask(question: str):
    used, note = choice, ""
    b = budget()
    if used == "anthropic" and b is not None and not b.allows():
        used = "fake"
        note = ("Today's demo budget for written answers is used up, so this answer quotes "
                "the best-matching passage instead. It resets at midnight UTC.")
    service = build_service(settings, embedder(), store(), llm(used),
                            reranker=reranker(), rewriter=rewriter(used))
    answer = service.ask(question)
    if used == "anthropic" and b is not None and answer.model:
        b.record(cost_usd(answer.model, answer.input_tokens, answer.output_tokens))
    return answer, note


for question, answer, note in st.session_state.history:
    with st.chat_message("user", avatar="👤"):
        st.markdown(question)
    with st.chat_message("assistant", avatar="🩺"):
        render(answer, note)

picked = None
if not st.session_state.history:
    st.markdown("<div class='muted' style='margin:1.2rem 0 .4rem'>TRY ONE OF THESE</div>",
                unsafe_allow_html=True)
    cols = st.columns(2)
    for i, ex in enumerate(EXAMPLES):
        if cols[i % 2].button(ex, key=f"ex{i}", use_container_width=True):
            picked = ex

question = st.chat_input("Ask about credentialing requirements...") or picked
if question:
    with st.chat_message("user", avatar="👤"):
        st.markdown(question)
    with st.chat_message("assistant", avatar="🩺"):
        try:
            with st.status("Searching the documents...", expanded=False) as status:
                answer, note = ask(question)
                status.update(label="Done", state="complete")
        except Exception as exc:  # show the type, not internals, on a public page
            st.error(f"Something went wrong ({type(exc).__name__}). Please try again.")
        else:
            render(answer, note)
            st.session_state.history.append((question, answer, note))
            if picked:
                st.rerun()  # hide the example buttons once the conversation has started
