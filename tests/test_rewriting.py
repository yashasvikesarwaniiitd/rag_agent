"""Query rewriting, entirely with fakes: no model download, no API call, $0."""
import json

from rag.adapters.embedders import HashingEmbedder
from rag.adapters.llms import ScriptedLLM
from rag.adapters.rerankers import FunctionReranker
from rag.adapters.rewriters import (REWRITE_SCHEMA, CachingRewriter, FunctionRewriter,
                                    LLMQueryRewriter)
from rag.adapters.stores import MemoryStore
from rag.application.answer import RagService
from rag.application.ingest import ingest
from rag.domain.models import Page
from rag.ports import LLMResult, Rewrite
from tests.conftest import ListLoader


# ---------------------------------------------------------------- the Haiku rewriter

def test_llm_rewriter_requests_json_and_cleans_the_reply():
    llm = ScriptedLLM([json.dumps({"queries": [
        "recredentialing frequency MA organization", "  ", "How often re-checked?",
        "recredentialing frequency MA organization", "b", "c", "d"]})])
    out = LLMQueryRewriter(llm).rewrite("How often re-checked?")
    # blanks, duplicates and an echo of the question itself are dropped; at most 3 kept
    assert out.queries == ["recredentialing frequency MA organization", "b", "c"]
    assert llm.calls[0]["json_schema"] == REWRITE_SCHEMA


# ---------------------------------------------------------------- the cache

class Billed:
    def __init__(self):
        self.calls = 0

    def rewrite(self, q):
        self.calls += 1
        return Rewrite(queries=[f"policy {q}"], input_tokens=300, output_tokens=100,
                       model="claude-haiku-5-5")


def test_cache_serves_repeat_questions_for_free(tmp_path):
    inner = Billed()
    c = CachingRewriter(inner, tmp_path / "r.sqlite", "haiku")
    first, second = c.rewrite("q"), c.rewrite("q")
    assert first.queries == second.queries == ["policy q"]
    assert inner.calls == 1 and second.input_tokens == 0
    assert (c.hits, c.misses, c.input_tokens, c.output_tokens) == (1, 1, 300, 100)


def test_cache_only_mode_never_calls_the_model(tmp_path):
    inner = Billed()
    c = CachingRewriter(inner, tmp_path / "r.sqlite", "haiku", paid=False)
    assert c.rewrite("not cached").queries == []
    assert inner.calls == 0 and c.unavailable == 1


def test_cache_only_mode_still_uses_what_a_paid_run_cached(tmp_path):
    CachingRewriter(Billed(), tmp_path / "r.sqlite", "haiku").rewrite("q")
    free = CachingRewriter(None, tmp_path / "r.sqlite", "haiku", paid=False)
    assert free.rewrite("q").queries == ["policy q"]


def test_a_different_model_does_not_reuse_rewrites(tmp_path):
    CachingRewriter(Billed(), tmp_path / "r.sqlite", "model-a").rewrite("q")
    other = CachingRewriter(None, tmp_path / "r.sqlite", "model-b", paid=False)
    assert other.rewrite("q").queries == [] and other.unavailable == 1


# ---------------------------------------------------------------- search with rewrites

PAGES = [Page(f"d{i}", f"d{i}.pdf", 1, f"Medicare enrollment revalidation filler {i}.")
         for i in range(6)]
PAGES.append(Page("reg", "reg.html", 1,
                  "MA organizations must recredential physicians at least every 3 years."))


def _store():
    store = MemoryStore()
    ingest(["x"], loader=ListLoader({"x": PAGES}), embedder=HashingEmbedder(), store=store)
    return store


PLAIN = "how often are network doctors re-checked"  # shares no words with the regulation


def test_rewrite_brings_in_a_chunk_the_plain_question_cannot_reach():
    store = _store()
    plain = RagService(embedder=HashingEmbedder(), store=store, llm=ScriptedLLM([]), top_k=3)
    assert "reg.html" not in [h.chunk.source for h in plain.retrieve(PLAIN)]
    rw = FunctionRewriter(lambda q: ["MA organizations recredential physicians frequency"])
    svc = RagService(embedder=HashingEmbedder(), store=store, llm=ScriptedLLM([]), top_k=3,
                     rewriter=rw)
    assert "reg.html" in [h.chunk.source for h in svc.retrieve(PLAIN)]


def test_off_topic_question_is_not_rewritten():
    rw = FunctionRewriter(lambda q: ["anything"])
    svc = RagService(embedder=HashingEmbedder(), store=_store(), llm=ScriptedLLM([]),
                     rewriter=rw, min_score=0.99)
    answer = svc.ask("What is the capital of France?")
    assert answer.refused and rw.calls == []


def test_reranker_scores_against_the_original_question():
    seen = []
    rr = FunctionReranker(lambda q, t: seen.append(q) or 0.0)
    svc = RagService(embedder=HashingEmbedder(), store=_store(), llm=ScriptedLLM([]),
                     reranker=rr, rewriter=FunctionRewriter(lambda q: ["policy words"]))
    svc.retrieve(PLAIN)
    assert set(seen) == {PLAIN}


def test_answer_reports_rewrites_and_includes_their_tokens():
    class TokenRewriter:
        def rewrite(self, q):
            return Rewrite(["MA organizations recredential physicians"], 300, 100)

    class Answering:
        def complete(self, **kw):
            return LLMResult(text="At least every 3 years [1].", input_tokens=2000,
                             output_tokens=200, model="claude-haiku-5-5")

    svc = RagService(embedder=HashingEmbedder(), store=_store(), llm=Answering(),
                     rewriter=TokenRewriter())
    answer = svc.ask(PLAIN)
    assert answer.rewrites == ["MA organizations recredential physicians"]
    assert (answer.input_tokens, answer.output_tokens) == (2300, 300)


def test_empty_rewrite_behaves_exactly_like_no_rewriter():
    store = _store()
    a = RagService(embedder=HashingEmbedder(), store=store, llm=ScriptedLLM([]), top_k=5)
    b = RagService(embedder=HashingEmbedder(), store=store, llm=ScriptedLLM([]), top_k=5,
                   rewriter=FunctionRewriter(lambda q: []))
    assert [h.chunk.id for h in a.retrieve(PLAIN)] == [h.chunk.id for h in b.retrieve(PLAIN)]
