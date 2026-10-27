from rag.adapters.embedders import HashingEmbedder
from rag.adapters.embedding_cache import CachingEmbedder
from rag.adapters.stores import MemoryStore
from rag.application.ingest import ingest
from rag.domain.models import Page
from tests.conftest import ListLoader


class CountingEmbedder(HashingEmbedder):
    def __init__(self):
        super().__init__()
        self.calls: list[list[str]] = []

    def embed_documents(self, texts):
        self.calls.append(list(texts))
        return super().embed_documents(texts)


def cache(tmp_path, inner=None, model="bge"):
    return CachingEmbedder(inner or CountingEmbedder(), tmp_path / "c.sqlite", model)


def test_second_pass_reuses_every_vector(tmp_path):
    c = cache(tmp_path)
    first = c.embed_documents(["a b", "c d"])
    second = c.embed_documents(["a b", "c d"])
    assert c.inner.calls == [["a b", "c d"]]
    assert all(abs(x - y) < 1e-6 for u, v in zip(first, second) for x, y in zip(u, v))
    assert (c.hits, c.misses) == (2, 2)


def test_only_new_text_is_embedded_and_order_is_kept(tmp_path):
    c = cache(tmp_path)
    c.embed_documents(["old one", "old two"])
    out = c.embed_documents(["old two", "brand new", "old one"])
    assert c.inner.calls[-1] == ["brand new"]
    expected = HashingEmbedder().embed_documents(["old two", "brand new", "old one"])
    assert all(abs(x - y) < 1e-6 for u, v in zip(out, expected) for x, y in zip(u, v))


def test_duplicates_in_one_call_are_embedded_once(tmp_path):
    c = cache(tmp_path)
    assert len(c.embed_documents(["same", "same", "same"])) == 3
    assert c.inner.calls == [["same"]]


def test_cache_survives_a_restart(tmp_path):
    cache(tmp_path).embed_documents(["persisted text"])
    again = cache(tmp_path)
    again.embed_documents(["persisted text"])
    assert again.inner.calls == []


def test_a_different_model_never_reuses_vectors(tmp_path):
    cache(tmp_path, model="bge-small").embed_documents(["text"])
    other = cache(tmp_path, model="other-model")
    other.embed_documents(["text"])
    assert other.inner.calls == [["text"]]


def test_queries_bypass_the_cache(tmp_path):
    c = cache(tmp_path)
    c.embed_query("q")
    c.embed_query("q")
    assert c.hits == c.misses == 0


def test_reingest_with_unchanged_text_embeds_nothing(tmp_path):
    inner = CountingEmbedder()
    c = cache(tmp_path, inner)
    loader = ListLoader({"d": [Page("doc", "doc.pdf", p, f"Page {p} text about credentialing.")
                               for p in (1, 2, 3)]})
    store = MemoryStore()
    ingest(["d"], loader=loader, embedder=c, store=store)
    ingest(["d"], loader=loader, embedder=c, store=store)
    assert len(inner.calls) == 1 and store.count() == 3


def test_ingest_embeds_a_document_in_one_call(tmp_path):
    inner = CountingEmbedder()
    pages = [Page("doc", "doc.pdf", p, f"Unique page {p} body text.") for p in range(1, 101)]
    ingest(["d"], loader=ListLoader({"d": pages}), embedder=inner, store=MemoryStore(),
           batch_size=10)
    assert len(inner.calls) == 1 and len(inner.calls[0]) == 100
