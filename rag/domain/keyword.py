"""Keyword search (BM25) and rank fusion. Pure functions, no dependencies.

Why keyword search at all: regulations are dense with exact terms ("recredential",
"438.214", "PAHP") that a small embedding model blurs together with nearby topics. In the
v0 measurement the regulation that answers "how often must MA plans recredential?" ranked
41st by vectors alone. BM25 rewards a chunk for containing the rare exact words of the
question, which is the signal vectors were missing.
"""
from __future__ import annotations

import math
import re
from collections import Counter

# Section numbers like 438.214 are kept whole; otherwise runs of letters/digits.
_TOKEN = re.compile(r"\d+(?:\.\d+)+|[a-z0-9]+")
_STOP = frozenset("""a an and are as at be by can do does for from has have how if in is it
its may must of on or should that the their this to was what when which who will with""".split())
_SUFFIXES = ("ations", "ation", "ings", "ing", "ies", "ed", "es", "s")


def stem(word: str) -> str:
    """Crude suffix stripping so recredentialed / recredentialing / recredential match.
    Deliberately conservative: only words longer than 4 letters, never digits."""
    if word.isdigit() or "." in word or len(word) <= 4:
        return word
    for suffix in _SUFFIXES:
        if word.endswith(suffix) and len(word) - len(suffix) >= 4:
            return word[: -len(suffix)] + ("y" if suffix == "ies" else "")
    return word


def tokenize(text: str) -> list[str]:
    return [stem(t) for t in _TOKEN.findall(text.lower()) if t not in _STOP]


class BM25:
    """Okapi BM25 over a fixed list of documents (here: chunk texts)."""

    def __init__(self, ids: list[str], texts: list[str], k1: float = 1.5, b: float = 0.75):
        self.ids = ids
        self.k1, self.b = k1, b
        self._tfs = [Counter(tokenize(t)) for t in texts]
        self._lens = [sum(tf.values()) for tf in self._tfs]
        self._avg = (sum(self._lens) / len(self._lens)) if self._lens else 0.0
        df: Counter = Counter()
        for tf in self._tfs:
            df.update(tf.keys())
        n = len(texts)
        self._idf = {term: math.log(1 + (n - d + 0.5) / (d + 0.5)) for term, d in df.items()}

    def __len__(self) -> int:
        return len(self.ids)

    def search(self, query: str, k: int) -> list[tuple[str, float]]:
        terms = [t for t in dict.fromkeys(tokenize(query)) if t in self._idf]
        if not terms:
            return []
        scored = []
        for i, tf in enumerate(self._tfs):
            score = 0.0
            norm = self.k1 * (1 - self.b + self.b * self._lens[i] / (self._avg or 1))
            for term in terms:
                f = tf.get(term)
                if f:
                    score += self._idf[term] * f * (self.k1 + 1) / (f + norm)
            if score > 0:
                scored.append((self.ids[i], score))
        scored.sort(key=lambda x: (-x[1], x[0]))
        return scored[:k]


def reciprocal_rank_fusion(rankings: list[list[str]], k: int = 60) -> list[tuple[str, float]]:
    """Merge ranked id lists: each list contributes 1/(k + rank). Uses ranks, not raw
    scores, so cosine similarities and BM25 scores never need to be put on one scale."""
    fused: dict[str, float] = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking, start=1):
            fused[item] = fused.get(item, 0.0) + 1.0 / (k + rank)
    return sorted(fused.items(), key=lambda x: (-x[1], x[0]))
