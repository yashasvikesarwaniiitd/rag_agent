"""Real PyMuPDF and real Chroma (both local and free). The sentence-transformers model is
NOT loaded here - that download belongs to the first real ingest, not to the test suite."""
from types import SimpleNamespace

import pymupdf
import pytest

from rag.adapters.embedders import HashingEmbedder
from rag.adapters.llms import AnthropicLLM
from rag.adapters.loaders import FileLoader, html_to_text
from rag.adapters.stores import ChromaStore
from rag.domain.models import Chunk


def make_pdf(path, pages):
    doc = pymupdf.open()
    for text in pages:
        doc.new_page().insert_text((72, 72), text)
    doc.save(path)
    doc.close()


def test_pdf_pages_are_one_based_and_keep_the_filename(tmp_path):
    pdf = tmp_path / "CMS Manual.pdf"
    make_pdf(pdf, ["first page text", "second page text"])
    pages = FileLoader().load(pdf)
    assert [(p.page, p.source, p.doc_id) for p in pages] == [
        (1, "CMS Manual.pdf", "cms_manual"), (2, "CMS Manual.pdf", "cms_manual")]
    assert "second page text" in pages[1].text


def test_html_loader_drops_scripts_and_navigation(tmp_path):
    html = tmp_path / "438.214.html"
    html.write_text("<nav>Menu</nav><script>x=1</script><h1>438.214</h1>"
                    "<p>Provider selection.</p>", encoding="utf-8")
    text = FileLoader().load(html)[0].text
    assert "Provider selection." in text and "Menu" not in text and "x=1" not in text


def test_html_block_tags_become_paragraph_breaks():
    assert "\n\n" in html_to_text("<p>a</p><p>b</p>")


def test_unsupported_file_type_is_rejected(tmp_path):
    bad = tmp_path / "x.docx"
    bad.write_bytes(b"")
    with pytest.raises(ValueError):
        FileLoader().load(bad)


def test_chroma_upsert_search_delete_and_persist(tmp_path):
    emb = HashingEmbedder()
    chunks = [Chunk("a:p1:c0", "a", "a.pdf", 1, "recredentialing every 36 months"),
              Chunk("b:p3:c0", "b", "b.pdf", 3, "DEA certificate for prescribers")]
    store = ChromaStore(tmp_path)
    store.upsert(chunks, emb.embed_documents([c.text for c in chunks]))

    reopened = ChromaStore(tmp_path)  # survives a restart
    assert reopened.count() == 2 and reopened.sources() == ["a.pdf", "b.pdf"]

    hits = reopened.search(emb.embed_query("DEA certificate"), k=2)
    assert hits[0].chunk == chunks[1]
    assert hits[0].score > hits[1].score

    reopened.delete_doc("b")
    assert reopened.count() == 1


def test_chroma_search_on_empty_store_returns_nothing(tmp_path):
    assert ChromaStore(tmp_path).search([0.1] * 256, k=5) == []


class FakeClient:
    """Stands in for anthropic.Anthropic so the adapter is tested without a network call."""

    def __init__(self, response):
        self.response = response
        self.kwargs = None
        self.messages = self

    def create(self, **kwargs):
        self.kwargs = kwargs
        return self.response


def test_anthropic_adapter_sends_cached_system_and_reads_usage():
    resp = SimpleNamespace(stop_reason="end_turn", model="claude-haiku-5-5",
                           content=[SimpleNamespace(type="text", text="ok [1]")],
                           usage=SimpleNamespace(input_tokens=120, output_tokens=8))
    client = FakeClient(resp)
    out = AnthropicLLM(client=client).complete(system="SYS", user="Q")
    assert out.text == "ok [1]" and out.input_tokens == 120 and out.output_tokens == 8
    assert client.kwargs["model"] == "claude-haiku-5-5"
    assert client.kwargs["system"][0]["cache_control"] == {"type": "ephemeral"}


def test_anthropic_adapter_raises_on_refusal_instead_of_returning_blank():
    resp = SimpleNamespace(stop_reason="refusal", content=[],
                           stop_details=SimpleNamespace(category="cyber"))
    with pytest.raises(RuntimeError, match="cyber"):
        AnthropicLLM(client=FakeClient(resp)).complete(system="S", user="Q")
