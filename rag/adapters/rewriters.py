"""Query rewriting: turn a plain-English question into the vocabulary the documents use.

Why: on the golden set every plain-English miss had its answer page in the candidate pool
but ranked outside the top 5, and the two worst ("how often are doctors re-checked?") use
words that never appear in the regulations. Searching with an added phrasing such as
"recredentialing frequency for MA organization network physicians" lets BM25 and the vectors
meet the documents halfway. The original question is always searched too, so a bad rewrite
can only add candidates, never take the original's away.

The prompt deliberately contains no facts from the golden set - only how to translate words
into regulatory vocabulary - so it cannot leak answers into retrieval.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Callable
from pathlib import Path

from ..ports import LLM, QueryRewriter, Rewrite

REWRITE_PROMPT_VERSION = "1"
REWRITE_SYSTEM = """You rewrite a user's question about US healthcare provider credentialing or \
enrollment into search queries that use the vocabulary of federal regulations and CMS manuals.

Rules:
- Return 1 to 3 queries. Each restates the SAME question; never answer it, never add facts.
- Replace everyday words with the terms a regulation or manual would use for the same idea.
- Keep the program the user named (Medicare, Medicare Advantage, Medicaid, Medicaid managed
  care) and the provider type. If the question is ambiguous between programs or processes,
  give one query per plausible reading.
- Keep each query under 20 words. No explanations."""

REWRITE_SCHEMA = {
    "type": "object",
    "properties": {"queries": {"type": "array", "items": {"type": "string"}}},
    "required": ["queries"],
    "additionalProperties": False,
}


class LLMQueryRewriter:
    def __init__(self, llm: LLM, max_queries: int = 3):
        self.llm = llm
        self.max_queries = max_queries

    def rewrite(self, question: str) -> Rewrite:
        result = self.llm.complete(system=REWRITE_SYSTEM, user=f"Question: {question}",
                                   max_tokens=2000, json_schema=REWRITE_SCHEMA)
        queries = json.loads(result.text).get("queries", [])
        clean = [q.strip() for q in queries if isinstance(q, str) and q.strip()]
        clean = [q for q in dict.fromkeys(clean) if q.lower() != question.strip().lower()]
        return Rewrite(queries=clean[: self.max_queries], input_tokens=result.input_tokens,
                       output_tokens=result.output_tokens, model=result.model)


class CachingRewriter:
    """Remembers rewrites by (model, prompt version, exact question), so retrieval experiments
    on the golden set can be repeated at $0 after one paid pass. A cached rewrite reports zero
    tokens: it cost nothing this time.

    paid=False makes it cache-only: a question with no cached rewrite is searched without
    one, and is counted in `unavailable`, instead of silently spending money."""

    def __init__(self, inner: QueryRewriter | None, path: str | Path, model_id: str,
                 paid: bool = True):
        self.inner = inner
        self.paid = paid and inner is not None
        self.model_id = model_id
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(path), check_same_thread=False)
        self._db.execute("CREATE TABLE IF NOT EXISTS rewrites (key TEXT PRIMARY KEY, q TEXT)")
        self.hits = self.misses = self.unavailable = 0
        self.input_tokens = self.output_tokens = 0  # spent by this instance on fresh rewrites

    def _key(self, question: str) -> str:
        raw = f"{self.model_id}\n{REWRITE_PROMPT_VERSION}\n{question.strip()}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def rewrite(self, question: str) -> Rewrite:
        key = self._key(question)
        row = self._db.execute("SELECT q FROM rewrites WHERE key = ?", (key,)).fetchone()
        if row:
            self.hits += 1
            return Rewrite(queries=json.loads(row[0]))
        if not self.paid:
            self.unavailable += 1
            return Rewrite(queries=[])
        fresh = self.inner.rewrite(question)
        self.misses += 1
        self.input_tokens += fresh.input_tokens
        self.output_tokens += fresh.output_tokens
        with self._db:
            self._db.execute("INSERT OR REPLACE INTO rewrites VALUES (?, ?)",
                             (key, json.dumps(fresh.queries)))
        return fresh


class FunctionRewriter:
    """Test fake."""

    def __init__(self, fn: Callable[[str], list[str]]):
        self.fn = fn
        self.calls: list[str] = []

    def rewrite(self, question: str) -> Rewrite:
        self.calls.append(question)
        return Rewrite(queries=self.fn(question))
