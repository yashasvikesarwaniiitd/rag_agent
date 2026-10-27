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
