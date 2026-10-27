"""End-to-end through RagService with fakes: no network, no model download, no API cost."""
from rag.adapters.embedders import HashingEmbedder
from rag.adapters.llms import ExtractiveLLM, ScriptedLLM
from rag.adapters.stores import MemoryStore
from rag.application.answer import RagService
from rag.application.ingest import ingest
from rag.application.prompts import SYSTEM
from rag.domain.models import Page
from tests.conftest import ListLoader


def service(store_pair, llm, top_k=3):
    embedder, store = store_pair
    return RagService(embedder=embedder, store=store, llm=llm, top_k=top_k)


def test_retrieval_ranks_the_matching_page_first(indexed_store):
    hits = service(indexed_store, ScriptedLLM([])).retrieve(
        "how often must providers be recredentialed")
    assert (hits[0].chunk.source, hits[0].chunk.page) == ("cms_manual.pdf", 1)


def test_cited_answer_maps_markers_to_source_and_page(indexed_store):
    llm = ScriptedLLM(lambda system, user: "At least every 36 months [1].")
    answer = service(indexed_store, llm).ask("How often must providers be recredentialed?")
    assert not answer.refused
    assert [(c.n, c.source, c.page) for c in answer.citations] == [(1, "cms_manual.pdf", 1)]


def test_prompt_sends_numbered_sources_and_the_question(indexed_store):
    llm = ScriptedLLM(["x [1]"])
    service(indexed_store, llm).ask("How often must providers be recredentialed?")
    call = llm.calls[0]
    assert call["system"] == SYSTEM
    assert '<source id="1" document="cms_manual.pdf" page="1">' in call["user"]
    assert call["user"].rstrip().endswith("Question: How often must providers be recredentialed?")


def test_not_found_is_reported_as_a_decline(indexed_store):
    answer = service(indexed_store, ScriptedLLM(["NOT_FOUND"])).ask("What is the capital of France?")
    assert answer.refused and "do not cover" in answer.reason
    assert answer.citations == []


def test_uncited_answer_is_never_shown(indexed_store):
    answer = service(indexed_store, ScriptedLLM(["Every 36 months."])).ask("recredentialing?")
    assert answer.refused and "no citations" in answer.reason
    assert answer.raw_model_text == "Every 36 months."  # kept for debugging, not displayed
    assert "36" not in answer.text


def test_invented_source_number_is_blocked(indexed_store):
    answer = service(indexed_store, ScriptedLLM(["Every 36 months [9]."])).ask("recredentialing?")
    assert answer.refused and "not retrieved" in answer.reason


def test_empty_index_declines_without_calling_the_model():
    llm = ScriptedLLM([])
    answer = RagService(embedder=HashingEmbedder(), store=MemoryStore(), llm=llm).ask("anything")
    assert answer.refused and answer.reason == "empty index"
    assert llm.calls == []


def test_blank_question_declines_without_calling_the_model(indexed_store):
    llm = ScriptedLLM([])
    assert service(indexed_store, llm).ask("   ").refused
    assert llm.calls == []


def test_extractive_fake_answers_from_the_top_source(indexed_store):
    answer = service(indexed_store, ExtractiveLLM()).ask("state license primary source verification")
    assert not answer.refused
    assert answer.citations[0].page == 2
    assert "licensing board" in answer.text


def test_reingesting_a_document_replaces_its_old_chunks():
    embedder, store = HashingEmbedder(), MemoryStore()
    v1 = [Page("doc", "doc.pdf", p, f"old text page {p}") for p in (1, 2, 3)]
    v2 = [Page("doc", "doc.pdf", 1, "new text")]
    ingest(["d"], loader=ListLoader({"d": v1}), embedder=embedder, store=store)
    ingest(["d"], loader=ListLoader({"d": v2}), embedder=embedder, store=store)
    assert store.count() == 1


def test_bad_file_is_skipped_and_reported():
    class Boom:
        def load(self, path):
            raise ValueError("corrupt")

    report = ingest(["bad.pdf"], loader=Boom(), embedder=HashingEmbedder(), store=MemoryStore())
    assert report.files == [] and report.skipped == ["bad.pdf: corrupt"]


def test_textless_document_is_flagged_as_possibly_scanned():
    loader = ListLoader({"scan.pdf": [Page("scan", "scan.pdf", 1, "  ")]})
    report = ingest(["scan.pdf"], loader=loader, embedder=HashingEmbedder(), store=MemoryStore())
    assert "scanned" in report.skipped[0] and report.empty_pages == 1
