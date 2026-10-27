import json
from pathlib import Path

from rag.domain.models import Chunk
from rag.domain.scoring import Expected, first_relevant_rank, is_relevant, summarise

GOLDEN = Path(__file__).resolve().parent.parent / "evals" / "golden.jsonl"


def chunk(doc_id, page):
    return Chunk(f"{doc_id}:p{page}:c0", doc_id, f"{doc_id}.pdf", page, "text")


def test_relevance_matches_doc_prefix_and_exact_page():
    exp = [Expected("ecfr_42cfr_422.204", 1)]
    assert is_relevant(chunk("ecfr_42cfr_422.204_ma_provider_selection", 1), exp)
    assert not is_relevant(chunk("ecfr_42cfr_422.204_ma_provider_selection", 2), exp)
    assert not is_relevant(chunk("ecfr_42cfr_438.214_medicaid", 1), exp)


def test_right_page_wrong_section_does_not_count():
    # Real case: Chapter 6 p4 holds section 20.2 (the answer) and section 30 (written notice).
    exp = [Expected("ch6", 4, contains="reviewed by clinical peers")]
    s30 = Chunk("ch6:p4:c2", "ch6", "ch6.pdf", 4, "Written notice of rules of participation.")
    s20 = Chunk("ch6:p4:c1", "ch6", "ch6.pdf", 4, "Standards should be  reviewed by Clinical\npeers.")
    assert not is_relevant(s30, exp)
    assert is_relevant(s20, exp)  # case and line breaks ignored


def test_contains_accepts_alternatives():
    exp = [Expected("ch6", 10, contains="other requirements | malpractice insurance meeting")]
    assert is_relevant(Chunk("x", "ch6", "x.pdf", 10, "Current malpractice insurance meeting MA"), exp)


def test_golden_phrases_are_not_blank():
    rows = [json.loads(l) for l in GOLDEN.read_text(encoding="utf-8").splitlines() if l]
    assert all(e.get("contains", "").strip() for r in rows for e in r["expected"])


def test_any_of_several_expected_pages_counts():
    exp = [Expected("a", 1), Expected("b", 13)]
    assert first_relevant_rank([chunk("x", 1), chunk("b", 13)], exp) == 2


def test_miss_is_none():
    assert first_relevant_rank([chunk("x", 1)], [Expected("a", 1)]) is None


def test_summary_counts_hits_within_k_and_mrr():
    s = summarise([1, 2, None, 7], k=5)
    assert (s.questions, s.hits) == (4, 2)
    assert abs(s.mrr - (1 + 0.5) / 4) < 1e-9


def test_golden_file_is_well_formed():
    rows = [json.loads(line) for line in GOLDEN.read_text(encoding="utf-8").splitlines() if line]
    ids = [r["id"] for r in rows]
    assert len(ids) == len(set(ids)), "duplicate ids"
    for r in rows:
        assert r["style"] in {"policy", "plain", "decline"}
        assert r["should_decline"] == (r["style"] == "decline")
        assert r["should_decline"] or r["expected"], f"{r['id']} has no expected page"
        for e in r["expected"]:
            Expected(**e)
