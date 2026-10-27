"""Question -> retrieve -> prompt -> model -> citation gate -> Answer."""
from __future__ import annotations

import time
from dataclasses import dataclass, field

from ..domain.citations import check_answer
from ..domain.keyword import BM25, reciprocal_rank_fusion
from ..domain.models import Answer, Citation, Hit
from ..ports import LLM, Embedder, QueryRewriter, Reranker, Rewrite, VectorStore
from .prompts import SYSTEM, build_user_prompt


@dataclass(frozen=True)
class SearchResult:
    hits: list[Hit]
    best: float
    rewrite: Rewrite = field(default_factory=lambda: Rewrite(queries=[]))


class RagService:
    """retrieval="hybrid" merges vector and BM25 keyword rankings; "vector" is v0 behaviour.

    min_score is the off-topic cutoff: if no chunk's vector similarity reaches it, the
    question is declined before any model call. It is checked on the VECTOR score because
    that is on a fixed 0-1 scale; BM25 scores grow with corpus size and can't be thresholded.

    reranker, when given, re-orders the first `rerank_pool` candidates (vector + keyword
    results merged) before the top_k are taken. On the golden set every answer was already
    in that pool; what was wrong was the order.

    rewriter, when given, adds policy-vocabulary phrasings of the question; each is searched
    alongside the original and all rankings are fused before re-ranking.
    """

    def __init__(self, *, embedder: Embedder, store: VectorStore, llm: LLM, top_k: int = 5,
                 max_tokens: int = 4000, retrieval: str = "hybrid", candidates: int = 30,
                 min_score: float = 0.0, reranker: Reranker | None = None,
                 rerank_pool: int = 25, rewriter: QueryRewriter | None = None):
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
        self.reranker = reranker
        self.rerank_pool = rerank_pool
        self.rewriter = rewriter
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

    def search_detailed(self, question: str) -> SearchResult:
        """Top-k hits, the best vector similarity of the ORIGINAL question (for the off-topic
        cutoff), and any rewrite used. A question already below the cutoff is not rewritten,
        so off-topic questions never cost a model call."""
        vector_hits = self.store.search(self.embedder.embed_query(question), self.candidates)
        best = max((h.score for h in vector_hits), default=0.0)

        rewrite = Rewrite(queries=[])
        if self.rewriter is not None and best >= self.min_score:
            rewrite = self.rewriter.rewrite(question)

        by_id = {h.chunk.id: h.chunk for h in vector_hits}
        rankings = [[h.chunk.id for h in vector_hits]]
        for extra in rewrite.queries:
            hits = self.store.search(self.embedder.embed_query(extra), self.candidates)
            by_id |= {h.chunk.id: h.chunk for h in hits}
            rankings.append([h.chunk.id for h in hits])
        if self.retrieval == "hybrid":
            keyword = self._keyword_index()
            by_id |= self._chunks
            for q in [question, *rewrite.queries]:
                rankings.append([cid for cid, _ in keyword.search(q, self.candidates)])

        if len(rankings) == 1:
            ranked = vector_hits
        else:
            ranked = [Hit(chunk=by_id[cid], score=score)
                      for cid, score in reciprocal_rank_fusion(rankings)]

        if self.reranker is not None and ranked:
            pool = ranked[: self.rerank_pool]
            # The re-ranker DOES get the section label, unlike the index: it reads question
            # and chunk together, so "60.3 Credentialing > Recredentialing" is evidence, not
            # noise. Measured: MRR 0.69 without the label, 0.81 with it (60-chunk pool).
            scores = self.reranker.score(question, [
                f"{h.chunk.context}\n{h.chunk.text}" if h.chunk.context else h.chunk.text
                for h in pool])
            # Stable on ties: equal scores keep the order the fast searches gave them.
            # Scored against the user's ORIGINAL question: the rewrites only widen the pool;
            # what has to be relevant is what was asked.
            order = sorted(range(len(pool)), key=lambda i: -scores[i])
            ranked = [Hit(chunk=pool[i].chunk, score=scores[i]) for i in order]
        return SearchResult(hits=ranked[: self.top_k], best=best, rewrite=rewrite)

    def search(self, question: str) -> tuple[list[Hit], float]:
        """Top-k hits plus the best vector similarity (for the off-topic cutoff)."""
        result = self.search_detailed(question)
        return result.hits, result.best

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

        found = self.search_detailed(question)
        hits, best, rewrite = found.hits, found.best, found.rewrite
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
        # Rewrite tokens are added in so the answer's cost covers both calls. Both use the
        # same model (RAG_MODEL), so one price applies.
        base = dict(hits=hits, raw_model_text=result.text, rewrites=rewrite.queries,
                    input_tokens=result.input_tokens + rewrite.input_tokens,
                    output_tokens=result.output_tokens + rewrite.output_tokens,
                    model=result.model)

        if result.stop_reason == "max_tokens":
            # Cut off mid-answer: a half sentence can still carry a valid [1] and pass the gate.
            return done(Answer(text="The answer was cut off before it finished.",
                               refused=True, reason="answer truncated (max_tokens)", **base))
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
