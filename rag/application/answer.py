"""Question -> retrieve -> prompt -> model -> citation gate -> Answer."""
from __future__ import annotations

import time

from ..domain.citations import check_answer
from ..domain.keyword import BM25, reciprocal_rank_fusion
from ..domain.models import Answer, Citation, Hit
from ..ports import LLM, Embedder, VectorStore
from .prompts import SYSTEM, build_user_prompt


class RagService:
    """retrieval="hybrid" merges vector and BM25 keyword rankings; "vector" is v0 behaviour.

    min_score is the off-topic cutoff: if no chunk's vector similarity reaches it, the
    question is declined before any model call. It is checked on the VECTOR score because
    that is on a fixed 0-1 scale; BM25 scores grow with corpus size and can't be thresholded.
    """

    def __init__(self, *, embedder: Embedder, store: VectorStore, llm: LLM, top_k: int = 5,
                 max_tokens: int = 1024, retrieval: str = "hybrid", candidates: int = 30,
                 min_score: float = 0.0):
        if retrieval not in {"hybrid", "vector"}:
            raise ValueError(f"retrieval must be 'hybrid' or 'vector', got {retrieval!r}")
        self.embedder = embedder
        self.store = store
        self.llm = llm
        self.top_k = top_k
        self.max_tokens = max_tokens
        self.retrieval = retrieval
        self.candidates = max(candidates, top_k)
        self.min_score = min_score
        self._bm25: BM25 | None = None
        self._chunks: dict = {}
        self._bm25_size = -1

    def _keyword_index(self) -> BM25:
        # Rebuilt from the store whenever its size changes (i.e. after an ingest), so the
        # keyword index can never drift from the vector index it is fused with.
        size = self.store.count()
        if self._bm25 is None or size != self._bm25_size:
            chunks = self.store.all_chunks()
            self._chunks = {c.id: c for c in chunks}
            self._bm25 = BM25([c.id for c in chunks], [c.text for c in chunks])
            self._bm25_size = size
        return self._bm25

    def search(self, question: str) -> tuple[list[Hit], float]:
        """Top-k hits plus the best vector similarity (for the off-topic cutoff)."""
        vector_hits = self.store.search(self.embedder.embed_query(question), self.candidates)
        best = max((h.score for h in vector_hits), default=0.0)
        if self.retrieval == "vector":
            return vector_hits[: self.top_k], best

        keyword_hits = self._keyword_index().search(question, self.candidates)
        fused = reciprocal_rank_fusion([[h.chunk.id for h in vector_hits],
                                        [cid for cid, _ in keyword_hits]])
        by_id = {h.chunk.id: h.chunk for h in vector_hits} | self._chunks
        return [Hit(chunk=by_id[cid], score=score) for cid, score in fused[: self.top_k]], best

    def retrieve(self, question: str) -> list[Hit]:
        return self.search(question)[0]

    def ask(self, question: str) -> Answer:
        started = time.perf_counter()

        def done(answer: Answer) -> Answer:
            answer.latency_ms = int((time.perf_counter() - started) * 1000)
            return answer

        question = question.strip()
        if not question:
            return done(Answer(text="Please ask a question.", refused=True, reason="empty question"))
        if self.store.count() == 0:
            return done(Answer(text="No documents are indexed yet. Run ingest first.",
                               refused=True, reason="empty index"))

        hits, best = self.search(question)
        if not hits:
            return done(Answer(text="Nothing relevant was found in the documents.",
                               refused=True, reason="no hits"))
        if best < self.min_score:
            return done(Answer(
                text="That doesn't look like something the indexed credentialing documents cover.",
                refused=True, hits=hits,
                reason=f"off-topic: best match {best:.2f} is below the {self.min_score:.2f} cutoff"))

        result = self.llm.complete(system=SYSTEM, user=build_user_prompt(question, hits),
                                   max_tokens=self.max_tokens)
        base = dict(hits=hits, raw_model_text=result.text, input_tokens=result.input_tokens,
                    output_tokens=result.output_tokens, model=result.model)

        gate = check_answer(result.text, len(hits))
        if gate.not_found:
            return done(Answer(text="The indexed documents don't answer this question.",
                               refused=True, reason=gate.reason, **base))
        if not gate.ok:
            # Fail closed: an uncited or wrongly-cited answer is never shown as an answer.
            return done(Answer(text="I couldn't produce an answer backed by the documents.",
                               refused=True, reason=gate.reason, **base))

        citations = [
            Citation(n=n, source=hits[n - 1].chunk.source, page=hits[n - 1].chunk.page,
                     chunk_id=hits[n - 1].chunk.id, snippet=hits[n - 1].chunk.text[:300],
                     context=hits[n - 1].chunk.context)
            for n in sorted(gate.cited)
        ]
        return done(Answer(text=result.text.strip(), citations=citations, **base))
