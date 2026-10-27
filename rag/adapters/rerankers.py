"""Re-rankers: re-order a candidate pool by reading question and chunk TOGETHER.

Vector search compares two separately-made summaries (question vector vs chunk vector), and
BM25 counts shared words. A cross-encoder reads the pair as one input, so it can tell that
"how often are doctors re-checked" is answered by "recredentialing at least every 3 years".
It is too slow to run over all chunks, which is why it only re-orders the ~60 candidates
the two fast searches already found.
"""
from __future__ import annotations

from collections.abc import Callable

DEFAULT_RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"  # ~90MB, Apache-2.0


class CrossEncoderReranker:
    def __init__(self, model_name: str = DEFAULT_RERANK_MODEL, batch_size: int = 32):
        from sentence_transformers import CrossEncoder

        self.model_name = model_name
        self.batch_size = batch_size
        self._model = CrossEncoder(model_name)

    def score(self, question: str, texts: list[str]) -> list[float]:
        if not texts:
            return []
        scores = self._model.predict([(question, t) for t in texts], batch_size=self.batch_size,
                                     show_progress_bar=False)
        return [float(s) for s in scores]


class FunctionReranker:
    """Test fake: scores each text with a plain function."""

    def __init__(self, fn: Callable[[str, str], float]):
        self.fn = fn
        self.calls: list[tuple[str, int]] = []

    def score(self, question: str, texts: list[str]) -> list[float]:
        self.calls.append((question, len(texts)))
        return [self.fn(question, t) for t in texts]
