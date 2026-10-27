import pytest

from rag.domain.chunking import chunk_page, chunk_pages, normalise
from rag.domain.models import Page


def page(text, n=1):
    return Page(doc_id="doc", source="doc.pdf", page=n, text=text)


def test_short_page_is_one_chunk_with_stable_id():
    chunks = chunk_page(page("Credentialing must be renewed every 36 months."))
    assert len(chunks) == 1
    assert chunks[0].id == "doc:p1:c0"
    assert chunks[0].page == 1 and chunks[0].source == "doc.pdf"


def test_empty_and_whitespace_pages_produce_no_chunks():
    assert chunk_page(page("   \n\n  \t ")) == []


def test_chunks_respect_max_chars():
    paras = "\n\n".join(f"Paragraph {i} " + "word " * 40 for i in range(20))
    chunks = chunk_page(page(paras), max_chars=500, overlap=50)
    assert len(chunks) > 1
    assert all(len(c.text) <= 500 for c in chunks)


def test_overlap_carries_text_into_next_chunk():
    paras = "\n\n".join(f"Section {i} says the board certification rule number {i}." * 3
                        for i in range(10))
    chunks = chunk_page(page(paras), max_chars=300, overlap=60)
    first_tail_words = chunks[0].text.split()[-3:]
    assert " ".join(first_tail_words) in chunks[1].text


def test_long_paragraph_without_breaks_is_hard_split():
    text = "The plan verifies licensure. " * 200  # one 5,800-char paragraph
    chunks = chunk_page(page(text), max_chars=800, overlap=100)
    assert len(chunks) > 5
    assert all(len(c.text) <= 800 for c in chunks)


def test_no_text_is_lost():
    words = [f"w{i}" for i in range(2000)]
    text = "\n\n".join(" ".join(words[i:i + 50]) for i in range(0, 2000, 50))
    chunks = chunk_page(page(text), max_chars=400, overlap=40)
    joined = " ".join(c.text for c in chunks)
    assert all(w in joined.split() for w in words)


def test_chunks_never_cross_pages():
    chunks = chunk_pages([page("Page one text.", 1), page("Page two text.", 2)])
    assert [(c.page, c.id) for c in chunks] == [(1, "doc:p1:c0"), (2, "doc:p2:c0")]


def test_overlap_must_be_smaller_than_max():
    with pytest.raises(ValueError):
        chunk_page(page("x"), max_chars=100, overlap=100)


def test_normalise_keeps_paragraph_breaks_and_collapses_spaces():
    assert normalise("a   b\r\n\r\n\r\n\r\nc") == "a b\n\nc"
