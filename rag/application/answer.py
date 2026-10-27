"""Question -> retrieve -> prompt -> model -> citation gate -> Answer."""
from __future__ import annotations

import time

from ..domain.citations import check_answer
from ..domain.models import Answer, Citation
from ..ports import LLM, Embedder, VectorStore
from .prompts import SYSTEM, build_user_prompt


class RagService:
    def __init__(self, *, embedder: Embedder, store: VectorStore, llm: LLM, top_k: int = 5,
                 max_tokens: int = 1024):
        self.embedder = embedder
        self.store = store
        self.llm = llm
        self.top_k = top_k
        self.max_tokens = max_tokens

    def retrieve(self, question: str):
        return self.store.search(self.embedder.embed_query(question), self.top_k)

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

        hits = self.retrieve(question)
        if not hits:
            return done(Answer(text="Nothing relevant was found in the documents.",
                               refused=True, reason="no hits"))

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
                     chunk_id=hits[n - 1].chunk.id, snippet=hits[n - 1].chunk.text[:300])
            for n in sorted(gate.cited)
        ]
        return done(Answer(text=result.text.strip(), citations=citations, **base))
